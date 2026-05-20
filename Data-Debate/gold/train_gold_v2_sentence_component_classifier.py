"""Train a gold_v2 sentence-level component classifier.

This model classifies one sentence/proposition at a time as:
- claim
- evidence
- background

It is intentionally separate from the token-level BIO component extractor so the
two approaches can be compared without changing the production pipeline.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
MODEL_NAME = "ytu-ce-cosmos/modernbert-tr-base-1k"
TOPIC_PATH = ROOT / "data" / "gold_400_topics.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "candidate" / "component_classifier"
MAX_LENGTH = 192
SEED = 42
ID2LABEL = {0: "claim", 1: "evidence", 2: "background"}
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
    topics = json.loads(TOPIC_PATH.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []
    for topic in topics:
        examples.append({"text": topic["claim"], "label": "claim", "topic": topic["topic"]})
        for key in ("support", "attack"):
            for evidence in topic[key]:
                examples.append({"text": evidence, "label": "evidence", "topic": topic["topic"]})
        for neutral in topic["neutral"]:
            examples.append({"text": neutral, "label": "background", "topic": topic["topic"]})
    return examples


def split_examples(examples: list[dict[str, str]], ratio: float = 0.2) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        by_label[example["label"]].append(example)

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for group in by_label.values():
        random.shuffle(group)
        n_val = max(1, int(len(group) * ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


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
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


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
    train_examples, val_examples = split_examples(examples)
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
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=encode(train_examples, tokenizer),
        eval_dataset=encode(val_examples, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
        class_weights=class_weights(train_examples),
    )
    trainer.train()

    final = OUTPUT_DIR / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved gold_v2 sentence component model to {final}")


if __name__ == "__main__":
    main()
