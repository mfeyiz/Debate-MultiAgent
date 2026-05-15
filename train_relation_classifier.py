"""Train ModernBERT for relation classification (support / attack / neutral).

Expects ``data/argument_dataset.json``.  Builds claim-evidence pairs:
* support  -> label 0
* attack   -> label 1
* neutral  -> label 2 (evidence not linked to a given claim)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from datasets import Dataset as HFDataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
DATASET_PATH = Path("data/argument_dataset.json")
OUTPUT_DIR = Path("models/relation_classifier")
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
SEED = 42

ID2LABEL = {0: "support", 1: "attack", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    return data


def build_pairs(raw: List[Dict]) -> List[Tuple[str, str, str]]:
    """Return list of (claim_text, evidence_text, label)."""
    pairs: List[Tuple[str, str, str]] = []
    for item in raw:
        data = item.get("data", item)
        claims = {c["id"]: c["text"] for c in data.get("claim", [])}
        evidences = {e["id"]: e["text"] for e in data.get("Evidence", [])}
        support_links = {(l["from"], l["to"]) for l in data.get("support", [])}
        attack_links = {(l["from"], l["to"]) for l in data.get("attack", [])}

        for cid, ctext in claims.items():
            for eid, etext in evidences.items():
                if (eid, cid) in support_links:
                    label = "support"
                elif (eid, cid) in attack_links:
                    label = "attack"
                else:
                    label = "neutral"
                pairs.append((ctext, etext, label))
    return pairs


def encode_pairs(pairs: List[Tuple[str, str, str]], tokenizer) -> HFDataset:
    texts = [f"{c} [SEP] {e}" for c, e, _ in pairs]
    labels = [LABEL2ID[l] for _, _, l in pairs]

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    entries = [
        {
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": labels[i],
        }
        for i in range(len(labels))
    ]
    return HFDataset.from_list(entries)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def freeze_base(model) -> None:
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n(model, n: int = 4) -> None:
    """Unfreeze last n transformer layers (works with BERT/RoBERTa/ModernBERT)."""
    base = getattr(model, "base_model", None) or getattr(model, "model", None)
    if base is None:
        raise AttributeError("Model has no 'base_model' or 'model' attribute")

    layers = None
    if hasattr(base, "layers"):
        layers = base.layers
    elif hasattr(base, "encoder"):
        enc = base.encoder
        if hasattr(enc, "layer"):
            layers = enc.layer
        elif hasattr(enc, "layers"):
            layers = enc.layers

    if layers is None:
        raise AttributeError("Could not find transformer layers in model")
    total = len(layers)
    for layer in layers[max(0, total - n) :]:
        for param in layer.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Place your 400-sample argument_dataset.json there."
        )

    raw = load_dataset(DATASET_PATH)
    pairs = build_pairs(raw)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = encode_pairs(train_pairs, tokenizer)
    val_ds = encode_pairs(val_pairs, tokenizer)
    datasets = DatasetDict({"train": train_ds, "validation": val_ds})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Phase 1
    print("Phase 1: training classification head only...")
    freeze_base(model)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "phase1"),
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # Phase 2
    print("Phase 2: unfreezing last 4 encoder layers...")
    unfreeze_last_n(model, n=4)
    training_args.output_dir = str(OUTPUT_DIR / "phase2")
    training_args.num_train_epochs = NUM_EPOCHS - 1

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # Save
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
