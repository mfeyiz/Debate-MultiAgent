"""Train a gold relation classifier from hand-authored relation pairs."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
DATASET_PATH = ROOT / "data" / "gold_relation_pairs.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gold" / "relation_classifier"
MAX_LENGTH = 256
SEED = 42

ID2LABEL = {0: "support", 1: "attack", 2: "neutral"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return the best local torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pairs() -> list[dict]:
    """Load hand-authored relation examples."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def stratified_split(pairs: list[dict], ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split examples while preserving labels."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_label[pair["label"]].append(pair)

    train: list[dict] = []
    val: list[dict] = []
    for label_pairs in by_label.values():
        random.shuffle(label_pairs)
        val_count = max(1, int(len(label_pairs) * ratio))
        val.extend(label_pairs[:val_count])
        train.extend(label_pairs[val_count:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def encode(pairs: list[dict], tokenizer) -> HFDataset:
    """Encode relation pairs for sequence classification."""
    texts = [
        f"{pair['claim_text']} [SEP] {pair['evidence_text']}"
        for pair in pairs
    ]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
    rows = [
        {
            "input_ids": enc["input_ids"][idx],
            "attention_mask": enc["attention_mask"][idx],
            "labels": LABEL2ID[pairs[idx]["label"]],
        }
        for idx in range(len(pairs))
    ]
    return HFDataset.from_list(rows)


def compute_metrics(eval_pred):
    """Compute macro relation metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> None:
    """Train and save the gold relation classifier."""
    set_seed()
    pairs = load_pairs()
    train_pairs, val_pairs = stratified_split(pairs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = encode(train_pairs, tokenizer)
    val_ds = encode(val_pairs, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(get_device())

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=18,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved gold relation model to {final_dir}")


if __name__ == "__main__":
    main()
