"""Train gold_v2 relation classifier from the static 400-example topic file."""

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

from quality_training_examples import relation_examples


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = os.getenv("MODEL_NAME", "ytu-ce-cosmos/modernbert-tr-base-1k")
TOPIC_PATH = ROOT / "data" / "gold_400_topics.json"
QUALITY_REGRESSION_PATH = ROOT / "data" / "quality_regression_examples.json"
V3_HARD_CASE_PATH = ROOT / "data" / "v3_hard_cases.json"
V4_RELATION_JSONL = ROOT / "data" / "v4" / "relation_pairs.jsonl"
V3_RELATION_JSONL = ROOT / "data" / "v3" / "relation_pairs.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "candidate" / "relation_classifier"
MAX_LENGTH = int(os.getenv("RELATION_MAX_LENGTH", "256"))
SEED = 42
ID2LABEL = {0: "support", 1: "attack", 2: "none"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def device() -> torch.device:
    if os.getenv("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pairs() -> list[dict]:
    if V4_RELATION_JSONL.exists():
        return [
            json.loads(line)
            for line in V4_RELATION_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if V3_RELATION_JSONL.exists():
        return [
            json.loads(line)
            for line in V3_RELATION_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    topics = json.loads(TOPIC_PATH.read_text(encoding="utf-8"))
    pairs: list[dict] = []
    for topic in topics:
        for label in ("support", "attack", "neutral"):
            for evidence in topic[label]:
                pairs.append(
                    {
                        "claim_text": topic["claim"],
                        "evidence_text": evidence,
                        "label": "none" if label == "neutral" else label,
                        "topic": topic["topic"],
                    }
                )
    if QUALITY_REGRESSION_PATH.exists():
        quality = json.loads(QUALITY_REGRESSION_PATH.read_text(encoding="utf-8"))
        pairs.extend(quality.get("relation_examples", []))
    if V3_HARD_CASE_PATH.exists():
        v3 = json.loads(V3_HARD_CASE_PATH.read_text(encoding="utf-8"))
        pairs.extend(v3.get("relation_examples", []))
    pairs.extend(relation_examples())
    for pair in pairs:
        if pair.get("label") == "neutral":
            pair["label"] = "none"
    return pairs


def split_pairs(
    pairs: list[dict],
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by topic to avoid same-topic relation leakage."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_topic[pair.get("topic", "unknown")].append(pair)

    topics = list(by_topic)
    random.shuffle(topics)
    n_test = max(1, int(len(topics) * test_ratio))
    n_val = max(1, int(len(topics) * validation_ratio))
    test_topics = set(topics[:n_test])
    val_topics = set(topics[n_test : n_test + n_val])

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
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


def encode(pairs: list[dict], tokenizer) -> HFDataset:
    texts = [f"{p['claim_text']} [SEP] {p['evidence_text']}" for p in pairs]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
    return HFDataset.from_list(
        [
            {
                "input_ids": enc["input_ids"][idx],
                "attention_mask": enc["attention_mask"][idx],
                "labels": LABEL2ID[pairs[idx]["label"]],
            }
            for idx in range(len(pairs))
        ]
    )


def metrics(eval_pred):
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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def apply_relation_thresholds(probs: np.ndarray, threshold: float) -> np.ndarray:
    raw_preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    preds = raw_preds.copy()
    none_id = LABEL2ID["none"]
    for idx, pred in enumerate(raw_preds):
        if ID2LABEL[int(pred)] in {"support", "attack"} and confidences[idx] < threshold:
            preds[idx] = none_id
    return preds


def calibrate_relation_thresholds(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Find a validation confidence threshold for graph-visible relations.

    The classifier itself remains a pure argmax model. The calibrated threshold is
    stored as artifact metadata so graph consumers can decide when to hide weak
    support/attack edges without pretending the model predicted a different class.
    """
    probs = softmax(logits)
    raw_preds = np.argmax(probs, axis=1)
    best = {
        "relation_confidence_threshold": 0.0,
        "macro_f1": f1_score(labels, raw_preds, average="macro", zero_division=0),
        "none_precision": precision_score(labels == LABEL2ID["none"], raw_preds == LABEL2ID["none"], zero_division=0),
        "visible_relation_recall": recall_score(
            labels != LABEL2ID["none"],
            raw_preds != LABEL2ID["none"],
            zero_division=0,
        ),
    }
    for threshold in np.arange(0.30, 0.91, 0.01):
        preds = apply_relation_thresholds(probs, float(threshold))
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        none_precision = precision_score(labels == LABEL2ID["none"], preds == LABEL2ID["none"], zero_division=0)
        visible_recall = recall_score(labels != LABEL2ID["none"], preds != LABEL2ID["none"], zero_division=0)
        if (macro_f1, none_precision, visible_recall) > (
            best["macro_f1"],
            best["none_precision"],
            best["visible_relation_recall"],
        ):
            best = {
                "relation_confidence_threshold": round(float(threshold), 2),
                "macro_f1": float(macro_f1),
                "none_precision": float(none_precision),
                "visible_relation_recall": float(visible_recall),
            }
    return best


class WeightedTrainer(Trainer):
    """Trainer with class weights for the none-heavy relation corpus."""

    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs: Any):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def class_weights(pairs: list[dict]) -> torch.Tensor:
    counts = Counter(pair["label"] for pair in pairs)
    total = sum(counts.values())
    weights = [
        total / (len(ID2LABEL) * max(1, counts[ID2LABEL[idx]]))
        for idx in range(len(ID2LABEL))
    ]
    return torch.tensor(weights, dtype=torch.float)


def main() -> None:
    seed_all()
    pairs = load_pairs()
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    save_checkpoints = os.getenv("SAVE_CHECKPOINTS", "0") == "1"
    print(f"Dataset counts: {Counter(pair['label'] for pair in pairs)}")
    print(f"Train counts: {Counter(pair['label'] for pair in train_pairs)}")
    print(f"Validation counts: {Counter(pair['label'] for pair in val_pairs)}")
    print(f"Test counts: {Counter(pair['label'] for pair in test_pairs)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device())
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
    train_dataset = encode(train_pairs, tokenizer)
    val_dataset = encode(val_pairs, tokenizer)
    test_dataset = encode(test_pairs, tokenizer)
    weights = class_weights(train_pairs)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
        class_weights=weights,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    val_predictions = trainer.predict(val_dataset)
    calibration = calibrate_relation_thresholds(
        np.asarray(val_predictions.predictions),
        np.asarray(val_predictions.label_ids),
    )
    artifact = OUTPUT_DIR / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact)
    tokenizer.save_pretrained(artifact)
    (artifact / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_name": MODEL_NAME,
        "output_dir": str(artifact),
        "dataset_version": "v4" if V4_RELATION_JSONL.exists() else "v3",
        "dataset_counts": dict(Counter(pair["label"] for pair in pairs)),
        "train_counts": dict(Counter(pair["label"] for pair in train_pairs)),
        "validation_counts": dict(Counter(pair["label"] for pair in val_pairs)),
        "test_counts": dict(Counter(pair["label"] for pair in test_pairs)),
        "split_strategy": "topic_group_holdout",
        "class_weights": {
            ID2LABEL[idx]: round(float(value), 6)
            for idx, value in enumerate(weights.tolist())
        },
        "calibration": calibration,
        "eval": eval_metrics,
        "test": test_metrics,
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved candidate relation model to {artifact}")


if __name__ == "__main__":
    main()
