"""Train ModernBERT for token-level component extraction (claim / evidence BIO).

Expects a JSON dataset at ``data/argument_dataset.json`` with the schema
used throughout this project.  Splits 90/10 train/val, freezes the base
encoder for the first epoch, then unfreezes the last 8 encoder layers
for fine-tuning.

Improvements:
- Exact character offset matching (no text.find fuzziness)
- Class weights to combat label imbalance
- CRF layer for valid BIO transition constraints
- Longer training (10 epochs)
- Unfreeze last 8 layers instead of 4
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torchcrf import CRF
from datasets import Dataset as HFDataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
DATASET_PATH = ROOT / "data" / "argument_dataset_fixed.json"
OUTPUT_DIR = ROOT.parent / "models" / "component_extractor"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
SEED = 42

LABEL_LIST = ["O", "B-CLAIM", "I-CLAIM", "B-EVIDENCE", "I-EVIDENCE"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

# Class weights: penalize O less, boost claim/evidence heavily
CLASS_WEIGHTS = torch.tensor([0.2, 2.0, 2.0, 2.0, 2.0])


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
# Dataset loading & BIO alignment with exact offsets
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    return data


def align_labels_with_exact_offsets(
    offsets: List[tuple[int, int]],
    text: str,
    spans: List[Dict[str, Any]],
) -> List[int]:
    """Map token char-offsets to BIO labels using EXACT offset matching."""
    labels = [-100 if start == end else 0 for start, end in offsets]

    for span in spans:
        span_text = span["text"]
        label_type = span["label"].upper()
        # Use exact start/end if available; otherwise fallback to find
        start = span.get("start", text.find(span_text))
        end = span.get("end", start + len(span_text) if start != -1 else -1)
        
        if start == -1 or end == -1:
            continue

        inside = False
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start or tok_start >= end:
                continue
            if tok_start == tok_end:
                continue
            # token overlaps span
            if not inside:
                labels[i] = LABEL2ID[f"B-{label_type}"]
                inside = True
            else:
                labels[i] = LABEL2ID[f"I-{label_type}"]
    return labels


def build_hf_dataset(raw: List[Dict[str, Any]], tokenizer) -> HFDataset:
    entries: List[Dict[str, Any]] = []
    for item in raw:
        data = item.get("data", item)
        text = data["text"]
        claim_spans = data.get("claim", [])
        evidence_spans = data.get("Evidence", [])
        all_spans = [{**c, "label": "claim"} for c in claim_spans] + [
            {**e, "label": "evidence"} for e in evidence_spans
        ]

        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        offsets = enc.offset_mapping
        labels = align_labels_with_exact_offsets(offsets, text, all_spans)
        # trim to actual length (after truncation)
        seq_len = len(enc.input_ids)
        labels = labels[:seq_len]
        entries.append(
            {
                "input_ids": enc.input_ids,
                "attention_mask": enc.attention_mask,
                "labels": labels,
            }
        )

    return HFDataset.from_list(entries)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def freeze_base(model) -> None:
    for param in model.base_model.parameters():
        param.requires_grad = False
    # classification head stays trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n(model, n: int = 8) -> None:
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
    for layer in layers[max(0, total - n):]:
        for param in layer.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Metrics (simple entity-level F1 via seqeval)
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    import numpy as np
    from seqeval.metrics import f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


# ---------------------------------------------------------------------------
# Custom Model with CRF
# ---------------------------------------------------------------------------
class ModernBERTCRF(nn.Module):
    """ModernBERT + CRF for token classification."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.backbone.classifier(sequence_output)

        mask = attention_mask.bool()

        if labels is not None:
            # Replace -100 with 0 for CRF (CRF doesn't handle -100)
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(logits, crf_labels, mask=mask, reduction="mean")
            return {"loss": loss, "logits": logits}
        else:
            predictions = self.crf.decode(logits, mask=mask)
            return {"logits": logits, "predictions": predictions}


# ---------------------------------------------------------------------------
# Custom Trainer with class weights
# ---------------------------------------------------------------------------
class WeightedCRFTrainer(Trainer):
    """Trainer that uses class weights + CRF loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            device = self.args.device
            self.class_weights = class_weights.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]

        if labels is not None:
            # Flatten for cross-entropy (token-level loss component)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Add CRF loss component if model has CRF
            if hasattr(model, "crf"):
                mask = inputs["attention_mask"].bool()
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                crf_loss = -model.crf(logits, crf_labels, mask=mask, reduction="mean")
                loss = loss + crf_loss

            outputs = {"loss": loss, "logits": logits}
            return (loss, outputs) if return_outputs else loss

        return outputs


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
            "Place your argument_dataset.json there."
        )

    raw = load_dataset(DATASET_PATH)
    print(f"Loaded {len(raw)} samples")
    random.shuffle(raw)
    split_idx = int(len(raw) * 0.9)
    train_raw, val_raw = raw[:split_idx], raw[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = build_hf_dataset(train_raw, tokenizer)
    val_ds = build_hf_dataset(val_raw, tokenizer)
    datasets = DatasetDict({"train": train_ds, "validation": val_ds})

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )

    # Initialize model with CRF
    model = ModernBERTCRF(MODEL_NAME, num_labels=len(LABEL_LIST))
    model.to(device)

    # Phase 1: freeze base, train head + CRF only
    print("Phase 1: training classification head + CRF only...")
    freeze_base(model.backbone)

    # Move class weights to device
    weights = CLASS_WEIGHTS.to(device)

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
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to=[],
    )

    trainer = WeightedCRFTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=weights,
    )
    trainer.train()

    # Phase 2: unfreeze last 8 encoder layers
    print("Phase 2: unfreezing last 8 encoder layers...")
    unfreeze_last_n(model.backbone, n=8)
    training_args.output_dir = str(OUTPUT_DIR / "phase2")
    training_args.num_train_epochs = NUM_EPOCHS - 1
    training_args.learning_rate = 1e-5  # lower LR for stable fine-tuning

    trainer = WeightedCRFTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=weights,
    )
    trainer.train()

    # Save final
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
