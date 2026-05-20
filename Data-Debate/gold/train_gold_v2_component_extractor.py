"""Train gold_v2 token-level component extractor from the static topic file."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
TOPIC_PATH = ROOT / "data" / "gold_400_topics.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gold_v2" / "component_extractor"
MAX_LENGTH = 512
SEED = 42
LABEL_LIST = ["O", "B-CLAIM", "I-CLAIM", "B-EVIDENCE", "I-EVIDENCE"]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
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


def span_item(item_id: int, segments: list[tuple[str, str]]) -> dict[str, Any]:
    text_parts: list[str] = []
    claims: list[dict[str, Any]] = []
    evidences: list[dict[str, Any]] = []
    cursor = 0
    claim_id = 1
    evidence_id = 1
    for idx, (label, segment) in enumerate(segments):
        if idx:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(segment)
        cursor += len(segment)
        end = cursor
        if label == "claim":
            claims.append({"id": f"c{claim_id}", "text": segment, "start": start, "end": end})
            claim_id += 1
        elif label == "evidence":
            evidences.append({"id": f"e{evidence_id}", "text": segment, "start": start, "end": end})
            evidence_id += 1
    text = "".join(text_parts)
    return {"id": item_id, "data": {"text": text, "claim": claims, "Evidence": evidences, "support": [], "attack": []}}


def component_items() -> list[dict[str, Any]]:
    topics = json.loads(TOPIC_PATH.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    item_id = 1
    for topic in topics:
        claim = topic["claim"]
        supports = topic["support"]
        attacks = topic["attack"]
        neutrals = topic["neutral"]
        layouts = [
            [("claim", claim), ("evidence", supports[0]), ("evidence", supports[1]), ("evidence", attacks[0]), ("background", neutrals[0])],
            [("background", neutrals[1]), ("claim", claim), ("evidence", attacks[1]), ("evidence", supports[2])],
            [("evidence", supports[3]), ("claim", claim), ("background", neutrals[2]), ("evidence", attacks[2])],
            [("claim", claim), ("background", neutrals[3]), ("evidence", supports[4]), ("evidence", attacks[3])],
            [("background", neutrals[4]), ("evidence", attacks[4]), ("claim", claim), ("evidence", supports[5])],
        ]
        for layout in layouts:
            items.append(span_item(item_id, layout))
            item_id += 1
    return items


def align(offsets: list[tuple[int, int]], spans: list[dict[str, Any]]) -> list[int]:
    labels = [-100 if start == end else LABEL2ID["O"] for start, end in offsets]
    for span in sorted(spans, key=lambda x: x["start"]):
        label_type = span["label"].upper()
        inside = False
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if labels[idx] == -100 or tok_end <= span["start"] or tok_start >= span["end"]:
                continue
            labels[idx] = LABEL2ID[f"{'I' if inside else 'B'}-{label_type}"]
            inside = True
    return labels


def build_dataset(raw: list[dict[str, Any]], tokenizer) -> HFDataset:
    rows: list[dict[str, Any]] = []
    for item in raw:
        data = item["data"]
        spans = [{**s, "label": "claim"} for s in data["claim"]] + [{**s, "label": "evidence"} for s in data["Evidence"]]
        enc = tokenizer(data["text"], truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True)
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": align(enc["offset_mapping"], spans)})
    return HFDataset.from_list(rows)


def metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[LABEL_LIST[p] for p, label in zip(pred, row) if label != -100] for pred, row in zip(predictions, labels)]
    true_labels = [[LABEL_LIST[label] for _, label in zip(pred, row) if label != -100] for pred, row in zip(predictions, labels)]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def main() -> None:
    seed_all()
    raw = component_items()
    random.shuffle(raw)
    split = max(1, int(len(raw) * 0.2))
    val_raw, train_raw = raw[:split], raw[split:]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device())
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
        train_dataset=build_dataset(train_raw, tokenizer),
        eval_dataset=build_dataset(val_raw, tokenizer),
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        compute_metrics=metrics,
    )
    trainer.train()
    final = OUTPUT_DIR / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved gold_v2 component model to {final}")


if __name__ == "__main__":
    main()
