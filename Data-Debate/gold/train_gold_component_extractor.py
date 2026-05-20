"""Train a token-level gold component extractor.

This intentionally uses plain token classification instead of CRF so the gold
baseline is stable and directly loadable by the existing runtime pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
DATASET_PATH = ROOT / "data" / "gold_component_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gold" / "component_extractor"
MAX_LENGTH = 512
SEED = 42

LABEL_LIST = ["O", "B-CLAIM", "I-CLAIM", "B-EVIDENCE", "I-EVIDENCE"]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
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


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load gold component examples."""
    return json.loads(path.read_text(encoding="utf-8"))


def align_labels(offsets: list[tuple[int, int]], spans: list[dict[str, Any]]) -> list[int]:
    """Align character spans to tokenizer offsets."""
    labels = [-100 if start == end else LABEL2ID["O"] for start, end in offsets]
    ordered_spans = sorted(spans, key=lambda span: span["start"])
    for span in ordered_spans:
        label_type = span["label"].upper()
        inside = False
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if labels[idx] == -100 or tok_end <= span["start"] or tok_start >= span["end"]:
                continue
            labels[idx] = LABEL2ID[f"{'I' if inside else 'B'}-{label_type}"]
            inside = True
    return labels


def build_dataset(raw: list[dict[str, Any]], tokenizer) -> HFDataset:
    """Build a Hugging Face token-classification dataset."""
    rows: list[dict[str, Any]] = []
    for item in raw:
        data = item["data"]
        text = data["text"]
        spans = [
            {**span, "label": "claim"}
            for span in data.get("claim", [])
        ] + [
            {**span, "label": "evidence"}
            for span in data.get("Evidence", [])
        ]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        labels = align_labels(enc["offset_mapping"], spans)
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        )
    return HFDataset.from_list(rows)


def compute_metrics(eval_pred):
    """Compute entity-level BIO metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [LABEL_LIST[p] for p, label in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[label] for _, label in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def main() -> None:
    """Train and save the gold component extractor."""
    set_seed()
    raw = load_dataset(DATASET_PATH)
    random.shuffle(raw)
    split = max(1, int(len(raw) * 0.2))
    val_raw = raw[:split]
    train_raw = raw[split:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = build_dataset(train_raw, tokenizer)
    val_ds = build_dataset(val_raw, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(get_device())

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=12,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=8e-6,
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
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved gold component model to {final_dir}")


if __name__ == "__main__":
    main()
