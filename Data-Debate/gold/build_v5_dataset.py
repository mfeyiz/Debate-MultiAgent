#!/usr/bin/env python3
"""Faz 3: assemble the v5 training dataset from consensus-labeled paragraphs.

Reads data/v5/labeled_paragraphs.jsonl (produced by scripts/consensus_label.py)
and explodes it into:
- data/v5/component_examples.jsonl  {text, label, topic, context, source, split}
- data/v5/relation_pairs.jsonl      {claim_text, evidence_text, label, topic, source, split}
- data/v5/audit.json

Leakage controls:
- GLOBAL text-level dedup: a component text (and a relation pair) appears at most
  once across the whole dataset, so no identical text can leak across train/val/test
  regardless of how the trainer splits by topic.
- Conflicting labels for the same normalized text/pair are DROPPED (treated as noise).
- An informational topic-based split field is attached (the trainers also re-split
  by topic at train time, which is the real leakage guard).

Public label schema is fixed: components claim/evidence/other, relations support/attack/none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V5_DIR = ROOT / "data" / "v5"
DEFAULT_IN = V5_DIR / "labeled_paragraphs.jsonl"

COMPONENT_LABELS = {"claim", "evidence", "other"}
RELATION_LABELS = {"support", "attack", "none"}


def normalize_text(text: str) -> str:
    """Aggressive normalization for cross-split duplicate detection."""
    text = str(text).strip().casefold()
    text = re.sub(r"[^\wığüşöçİĞÜŞÖÇ ]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def split_for_topic(topic: str, test_pct: int = 15, val_pct: int = 15) -> str:
    """Deterministic topic-based split bucket (informational)."""
    digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "validation"
    return "train"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def dedup_components(paragraphs: list[dict]) -> tuple[list[dict], Counter]:
    """One row per unique normalized text; drop texts with conflicting labels."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for p in paragraphs:
        for c in p["components"]:
            if c["label"] not in COMPONENT_LABELS:
                continue
            row = {
                "text": c["text"],
                "label": c["label"],
                "topic": p["topic"],
                "context": p["text"],
                "source": "v5_consensus",
                "generator": p.get("generator"),
            }
            by_key[normalize_text(c["text"])].append(row)

    audit = Counter()
    out: list[dict] = []
    for group in by_key.values():
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            audit["dropped_conflicting_text"] += 1
            continue
        row = group[0]
        row["split"] = split_for_topic(row["topic"])
        out.append(row)
        audit[f"label:{row['label']}"] += 1
    audit["unique_texts"] = len(by_key)
    audit["kept"] = len(out)
    return out, audit


def dedup_relations(paragraphs: list[dict]) -> tuple[list[dict], Counter]:
    """One row per unique (claim, evidence) normalized pair; drop conflicts.

    Convention (matches the relation classifier input ``{claim} [SEP] {evidence}``):
    claim_text = relation target text, evidence_text = relation source text.
    """
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in paragraphs:
        cid_text = {c["id"]: c["text"] for c in p["components"]}
        for r in p["relations"]:
            if r["label"] not in RELATION_LABELS:
                continue
            src_text = cid_text.get(r["from"])
            tgt_text = cid_text.get(r["to"])
            if not src_text or not tgt_text:
                continue
            row = {
                "claim_text": tgt_text,
                "evidence_text": src_text,
                "label": r["label"],
                "topic": p["topic"],
                "source": "v5_consensus",
                "generator": p.get("generator"),
            }
            by_key[(normalize_text(tgt_text), normalize_text(src_text))].append(row)

    audit = Counter()
    out: list[dict] = []
    for group in by_key.values():
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            audit["dropped_conflicting_pair"] += 1
            continue
        row = group[0]
        row["split"] = split_for_topic(row["topic"])
        out.append(row)
        audit[f"label:{row['label']}"] += 1
    audit["unique_pairs"] = len(by_key)
    audit["kept"] = len(out)
    return out, audit


def audit_rows(rows: list[dict], label_field: str) -> dict:
    return {
        "total": len(rows),
        "labels": dict(Counter(r[label_field] for r in rows)),
        "splits": dict(Counter(r.get("split", "train") for r in rows)),
        "generators": dict(Counter(r.get("generator", "?") for r in rows)),
    }


def check_split_leakage(rows: list[dict], text_fields: tuple[str, ...]) -> int:
    """Count normalized texts that appear in more than one split (should be 0)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        key = "|".join(normalize_text(r[f]) for f in text_fields)
        seen[key].add(r.get("split", "train"))
    return sum(1 for splits in seen.values() if len(splits) > 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    args = parser.parse_args()

    paragraphs = read_jsonl(args.input)
    if not paragraphs:
        raise SystemExit(f"HATA: girdi boş: {args.input}")

    components, comp_audit = dedup_components(paragraphs)
    relations, rel_audit = dedup_relations(paragraphs)

    write_jsonl(V5_DIR / "component_examples.jsonl", components)
    write_jsonl(V5_DIR / "relation_pairs.jsonl", relations)

    comp_leak = check_split_leakage(components, ("text",))
    rel_leak = check_split_leakage(relations, ("claim_text", "evidence_text"))

    audit = {
        "version": "v5",
        "source_paragraphs": len(paragraphs),
        "components": audit_rows(components, "label"),
        "relations": audit_rows(relations, "label"),
        "component_dedup": dict(comp_audit),
        "relation_dedup": dict(rel_audit),
        "leakage_texts_in_multiple_splits": {"components": comp_leak, "relations": rel_leak},
        "bad_component_labels": sorted({r["label"] for r in components if r["label"] not in COMPONENT_LABELS}),
        "bad_relation_labels": sorted({r["label"] for r in relations if r["label"] not in RELATION_LABELS}),
    }
    (V5_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["bad_component_labels"] or audit["bad_relation_labels"]:
        raise SystemExit("HATA: geçersiz etiket bulundu.")


if __name__ == "__main__":
    main()
