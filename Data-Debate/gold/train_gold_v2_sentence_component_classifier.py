"""Train a gold_v2 sentence-level component classifier.

This model classifies one sentence/proposition at a time as:
- claim
- evidence
- other

It is intentionally separate from the token-level BIO component extractor so the
two approaches can be compared without changing the production pipeline.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback

from quality_training_examples import component_examples


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = os.getenv("MODEL_NAME", "ytu-ce-cosmos/modernbert-tr-base-1k")
TOPIC_PATH = ROOT / "data" / "gold_400_topics.json"
QUALITY_REGRESSION_PATH = ROOT / "data" / "quality_regression_examples.json"
V3_HARD_CASE_PATH = ROOT / "data" / "v3_hard_cases.json"
V5_COMPONENT_JSONL = ROOT / "data" / "v5" / "component_examples.jsonl"
V4_COMPONENT_JSONL = ROOT / "data" / "v4" / "component_examples.jsonl"
V3_COMPONENT_JSONL = ROOT / "data" / "v3" / "component_examples.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "candidate" / "component_classifier"
MAX_LENGTH = int(os.getenv("COMPONENT_MAX_LENGTH", "288"))
# Context-aware input: classify the target sentence together with its paragraph.
# This lets the model use discourse role (claim vs evidence often depends on context).
CONTEXT_AWARE = os.getenv("COMPONENT_CONTEXT_AWARE", "1") == "1"
SEED = int(os.getenv("SEED", "42"))
# v8 scheme: the support class is named "premise" (was "evidence" in v5–v7).
ID2LABEL = {0: "claim", 1: "premise", 2: "other"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

# Map legacy/source labels onto the v8 scheme so existing v5 data trains as-is.
LABEL_REMAP = {"evidence": "premise", "background": "other"}


def _remap_labels(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    for example in examples:
        label = example.get("label")
        if label in LABEL_REMAP:
            example["label"] = LABEL_REMAP[label]
    return examples


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_examples() -> list[dict[str, str]]:
    for path in (V5_COMPONENT_JSONL, V4_COMPONENT_JSONL, V3_COMPONENT_JSONL):
        if path.exists():
            return _remap_labels([
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ])

    topics = json.loads(TOPIC_PATH.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []
    for topic in topics:
        examples.append({"text": topic["claim"], "label": "claim", "topic": topic["topic"]})
        for key in ("support", "attack"):
            for evidence in topic[key]:
                examples.append({"text": evidence, "label": "evidence", "topic": topic["topic"]})
        for neutral in topic["neutral"]:
                examples.append({"text": neutral, "label": "other", "topic": topic["topic"]})
    if QUALITY_REGRESSION_PATH.exists():
        quality = json.loads(QUALITY_REGRESSION_PATH.read_text(encoding="utf-8"))
        examples.extend(quality.get("component_examples", []))
    if V3_HARD_CASE_PATH.exists():
        v3 = json.loads(V3_HARD_CASE_PATH.read_text(encoding="utf-8"))
        examples.extend(v3.get("component_examples", []))
    examples.extend(component_examples())
    return _remap_labels(examples)


def split_examples(
    examples: list[dict[str, str]],
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split by topic so near-duplicate topic families do not leak across sets."""
    by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        by_topic[example.get("topic", "unknown")].append(example)

    topics = list(by_topic)
    random.shuffle(topics)
    n_test = max(1, int(len(topics) * test_ratio))
    n_val = max(1, int(len(topics) * validation_ratio))
    test_topics = set(topics[:n_test])
    val_topics = set(topics[n_test : n_test + n_val])

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for topic, group in by_topic.items():
        if topic in test_topics:
            test.extend(group)
        elif topic in val_topics:
            val.extend(group)
        else:
            train.extend(group)

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def encode(examples: list[dict[str, str]], tokenizer) -> HFDataset:
    texts = [example["text"] for example in examples]
    contexts = [example.get("context") for example in examples]
    if CONTEXT_AWARE and any(contexts):
        # Pair encoding: [CLS] target [SEP] paragraph-context [SEP].
        # truncation="only_second" keeps the full target sentence, trims context.
        enc = tokenizer(
            texts,
            [ctx or txt for ctx, txt in zip(contexts, texts)],
            truncation="only_second",
            max_length=MAX_LENGTH,
            padding=False,
        )
    else:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
    return HFDataset.from_list(
        [
            {
                "input_ids": enc["input_ids"][idx],
                "attention_mask": enc["attention_mask"][idx],
                "labels": LABEL2ID[examples[idx]["label"]],
            }
            for idx in range(len(examples))
        ]
    )


def metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    result = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }
    for label_id, label_name in ID2LABEL.items():
        result[f"{label_name}_recall"] = recall_score(
            labels == label_id,
            preds == label_id,
            zero_division=0,
        )
        result[f"{label_name}_precision"] = precision_score(
            labels == label_id,
            preds == label_id,
            zero_division=0,
        )
    return result


class WeightedTrainer(Trainer):
    """Trainer with class weights for the underrepresented claim class."""

    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs: Any):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def class_weights(examples: list[dict[str, str]]) -> torch.Tensor:
    counts = Counter(example["label"] for example in examples)
    total = sum(counts.values())
    weights = [
        total / (len(ID2LABEL) * max(1, counts[ID2LABEL[idx]]))
        for idx in range(len(ID2LABEL))
    ]
    return torch.tensor(weights, dtype=torch.float)


def freeze_lower_layers(model, num_unfrozen: int = 6) -> None:
    """Freeze all transformer layers except the top `num_unfrozen`.

    This reduces overfitting when training data is limited.
    """
    # Freeze embeddings
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False
    # Freeze lower encoder layers
    encoder_layers = model.base_model.layers
    num_layers = len(encoder_layers)
    freeze_until = max(0, num_layers - num_unfrozen)
    for layer_idx in range(freeze_until):
        for param in encoder_layers[layer_idx].parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Froze {freeze_until}/{num_layers} layers. Trainable: {trainable:,}/{total:,} params ({100*trainable/total:.1f}%)")


def main() -> None:
    seed_all()
    examples = load_examples()
    train_examples, val_examples, test_examples = split_examples(examples)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        mlp_dropout=float(os.getenv("HIDDEN_DROPOUT", "0.2")),
        attention_dropout=float(os.getenv("ATTN_DROPOUT", "0.15")),
        classifier_dropout=float(os.getenv("CLASSIFIER_DROPOUT", "0.3")),
    )
    # Freeze lower layers for better generalization
    num_unfrozen = int(os.getenv("NUM_UNFROZEN_LAYERS", "6"))
    freeze_lower_layers(model, num_unfrozen)
    model.to(device())

    print(f"Dataset counts: {Counter(example['label'] for example in examples)}")
    print(f"Train counts: {Counter(example['label'] for example in train_examples)}")
    print(f"Validation counts: {Counter(example['label'] for example in val_examples)}")
    print(f"Test counts: {Counter(example['label'] for example in test_examples)}")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=float(os.getenv("TRAIN_EPOCHS", "12")),
        per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "16")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "32")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "2")),
        learning_rate=float(os.getenv("TRAIN_LR", "2e-5")),
        weight_decay=0.01,
        warmup_ratio=float(os.getenv("WARMUP_RATIO", "0.1")),
        lr_scheduler_type=os.getenv("LR_SCHEDULER", "cosine"),
        label_smoothing_factor=float(os.getenv("LABEL_SMOOTHING", "0.1")),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        seed=SEED,
        report_to=[],
    )
    weights = class_weights(train_examples)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=encode(train_examples, tokenizer),
        eval_dataset=encode(val_examples, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(encode(test_examples, tokenizer), metric_key_prefix="test")
    # Train-set metrics expose the overfitting gap (train_f1 - test_f1).
    # Sample to bound memory/time on low-RAM machines; a sample estimates the gap well.
    train_sample = train_examples if len(train_examples) <= 800 else random.sample(train_examples, 800)
    train_metrics = trainer.evaluate(encode(train_sample, tokenizer), metric_key_prefix="train")
    overfitting_gap = round(
        float(train_metrics.get("train_f1", 0.0)) - float(test_metrics.get("test_f1", 0.0)), 4
    )
    print(
        f"Overfitting gap (train_f1 - test_f1): {overfitting_gap} "
        f"(train_f1={train_metrics.get('train_f1'):.4f}, test_f1={test_metrics.get('test_f1'):.4f})"
    )

    artifact = OUTPUT_DIR / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact)
    tokenizer.save_pretrained(artifact)
    context_aware_flag = bool(CONTEXT_AWARE and any(e.get("context") for e in examples))
    # Serve-time marker so bert_service feeds the same (target, context) pair input.
    (artifact / "component_meta.json").write_text(
        json.dumps({"context_aware": context_aware_flag, "max_length": MAX_LENGTH}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_name": MODEL_NAME,
        "output_dir": str(artifact),
        "dataset_version": "v8",
        "label_scheme": list(ID2LABEL.values()),
        "context_aware": context_aware_flag,
        "seed": SEED,
        "dataset_counts": dict(Counter(example["label"] for example in examples)),
        "train_counts": dict(Counter(example["label"] for example in train_examples)),
        "validation_counts": dict(Counter(example["label"] for example in val_examples)),
        "test_counts": dict(Counter(example["label"] for example in test_examples)),
        "split_strategy": "topic_group_holdout",
        "class_weights": {
            ID2LABEL[idx]: round(float(value), 6)
            for idx, value in enumerate(weights.tolist())
        },
        "eval": eval_metrics,
        "test": test_metrics,
        "train": train_metrics,
        "overfitting_gap": overfitting_gap,
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved candidate sentence component model to {artifact}")


if __name__ == "__main__":
    main()
