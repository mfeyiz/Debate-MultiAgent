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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from quality_training_examples import component_examples


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = os.getenv("MODEL_NAME", "ytu-ce-cosmos/modernbert-tr-base-1k")
TOPIC_PATH = ROOT / "data" / "gold_400_topics.json"
QUALITY_REGRESSION_PATH = ROOT / "data" / "quality_regression_examples.json"
V3_HARD_CASE_PATH = ROOT / "data" / "v3_hard_cases.json"
V4_COMPONENT_JSONL = ROOT / "data" / "v4" / "component_examples.jsonl"
V3_COMPONENT_JSONL = ROOT / "data" / "v3" / "component_examples.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "candidate" / "component_classifier"
MAX_LENGTH = 192
SEED = 42
ID2LABEL = {0: "claim", 1: "evidence", 2: "other"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


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
    if V4_COMPONENT_JSONL.exists():
        return [
            json.loads(line)
            for line in V4_COMPONENT_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if V3_COMPONENT_JSONL.exists():
        return [
            json.loads(line)
            for line in V3_COMPONENT_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

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
    for example in examples:
        if example.get("label") == "background":
            example["label"] = "other"
    return examples


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
    enc = tokenizer(
        [example["text"] for example in examples],
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


def main() -> None:
    seed_all()
    examples = load_examples()
    train_examples, val_examples, test_examples = split_examples(examples)
    save_checkpoints = os.getenv("SAVE_CHECKPOINTS", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device())

    print(f"Dataset counts: {Counter(example['label'] for example in examples)}")
    print(f"Train counts: {Counter(example['label'] for example in train_examples)}")
    print(f"Validation counts: {Counter(example['label'] for example in val_examples)}")
    print(f"Test counts: {Counter(example['label'] for example in test_examples)}")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=float(os.getenv("TRAIN_EPOCHS", "8")),
        per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "8")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "8")),
        learning_rate=float(os.getenv("TRAIN_LR", "8e-6")),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch" if save_checkpoints else "no",
        save_total_limit=1,
        logging_steps=10,
        load_best_model_at_end=save_checkpoints,
        metric_for_best_model="f1",
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
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(encode(test_examples, tokenizer), metric_key_prefix="test")

    artifact = OUTPUT_DIR / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact)
    tokenizer.save_pretrained(artifact)
    summary = {
        "model_name": MODEL_NAME,
        "output_dir": str(artifact),
        "dataset_version": "v4" if V4_COMPONENT_JSONL.exists() else "v3",
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
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved candidate sentence component model to {artifact}")


if __name__ == "__main__":
    main()
