"""Train gold_v2 relation classifier from the static 400-example topic file."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
TOPIC_PATH = ROOT / "data" / "gold_400_topics.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gold_v2" / "relation_classifier"
MAX_LENGTH = 256
SEED = 42
ID2LABEL = {0: "support", 1: "attack", 2: "neutral"}
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


def load_pairs() -> list[dict]:
    topics = json.loads(TOPIC_PATH.read_text(encoding="utf-8"))
    pairs: list[dict] = []
    for topic in topics:
        for label in ("support", "attack", "neutral"):
            for evidence in topic[label]:
                pairs.append(
                    {
                        "claim_text": topic["claim"],
                        "evidence_text": evidence,
                        "label": label,
                        "topic": topic["topic"],
                    }
                )
    return pairs


def split_pairs(pairs: list[dict], ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_label[pair["label"]].append(pair)
    train: list[dict] = []
    val: list[dict] = []
    for group in by_label.values():
        random.shuffle(group)
        n_val = max(1, int(len(group) * ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


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
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> None:
    seed_all()
    pairs = load_pairs()
    train_pairs, val_pairs = split_pairs(pairs)
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
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=8e-6,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
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
        train_dataset=encode(train_pairs, tokenizer),
        eval_dataset=encode(val_pairs, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    final = OUTPUT_DIR / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved gold_v2 relation model to {final}")


if __name__ == "__main__":
    main()
