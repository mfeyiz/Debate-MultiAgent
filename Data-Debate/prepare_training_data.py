"""Prepare corrected argument-mining training datasets.

The source annotation file is useful for component extraction, but the original
relation trainer never produced neutral pairs because each sample usually has a
single claim and every evidence span is linked to that claim. This script keeps
the valid span annotations and creates an explicit, balanced relation-pair file
with cross-topic neutral examples.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SOURCE_DATASET = DATA_DIR / "argument_dataset.json"
COMPONENT_DATASET = DATA_DIR / "argument_dataset_fixed.json"
RELATION_DATASET = DATA_DIR / "relation_pairs_fixed.json"
SEED = 42

TURKISH_STOPWORDS = {
    "acaba",
    "ama",
    "ancak",
    "artık",
    "az",
    "bazı",
    "belki",
    "ben",
    "bile",
    "bir",
    "birçok",
    "biri",
    "biz",
    "bu",
    "buna",
    "bunda",
    "bundan",
    "bunlar",
    "bunun",
    "çok",
    "çünkü",
    "da",
    "daha",
    "de",
    "defa",
    "diye",
    "eğer",
    "en",
    "fakat",
    "gibi",
    "hem",
    "hep",
    "hepsi",
    "her",
    "hiç",
    "için",
    "ile",
    "ise",
    "kez",
    "ki",
    "kim",
    "mı",
    "mu",
    "mü",
    "nasıl",
    "ne",
    "neden",
    "nerde",
    "nerede",
    "nereye",
    "niçin",
    "niye",
    "o",
    "sanki",
    "şey",
    "siz",
    "şu",
    "tüm",
    "ve",
    "veya",
    "ya",
    "yani",
}


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a project dataset JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data", data.get("samples", []))
    return data


def dump_json(path: Path, data: Any) -> None:
    """Write deterministic UTF-8 JSON."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def valid_span(text: str, span: dict[str, Any]) -> bool:
    """Return whether a span has exact, usable offsets."""
    if not span.get("text"):
        return False
    start = span.get("start")
    end = span.get("end")
    return isinstance(start, int) and isinstance(end, int) and text[start:end] == span["text"]


def normalize_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Keep only samples with at least one valid claim and one valid evidence."""
    data = item.get("data", item)
    text = data.get("text", "")
    claims = [span for span in data.get("claim", []) if valid_span(text, span)]
    evidences = [span for span in data.get("Evidence", []) if valid_span(text, span)]
    if not text or not claims or not evidences:
        return None

    valid_ids = {span["id"] for span in [*claims, *evidences]}
    support = [
        link
        for link in data.get("support", [])
        if link.get("from") in valid_ids and link.get("to") in valid_ids
    ]
    attack = [
        link
        for link in data.get("attack", [])
        if link.get("from") in valid_ids and link.get("to") in valid_ids
    ]

    return {
        "id": item.get("id"),
        "data": {
            "text": text,
            "claim": claims,
            "Evidence": evidences,
            "support": support,
            "attack": attack,
        },
    }


def content_tokens(text: str) -> set[str]:
    """Extract coarse Turkish content tokens for neutral-pair filtering."""
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9%]+", text.lower())
    return {token for token in tokens if len(token) >= 4 and token not in TURKISH_STOPWORDS}


def jaccard(left: str, right: str) -> float:
    """Compute a simple lexical overlap score."""
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def linked_pairs(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build positive support and attack relation pairs from span links."""
    support_pairs: list[dict[str, Any]] = []
    attack_pairs: list[dict[str, Any]] = []

    for item in items:
        data = item["data"]
        claims = {span["id"]: span["text"] for span in data.get("claim", [])}
        evidences = {span["id"]: span["text"] for span in data.get("Evidence", [])}

        for link in data.get("support", []):
            claim_text = claims.get(link.get("to"))
            evidence_text = evidences.get(link.get("from"))
            if claim_text and evidence_text:
                support_pairs.append(
                    {
                        "claim_text": claim_text,
                        "evidence_text": evidence_text,
                        "label": "support",
                        "claim_item_id": item["id"],
                        "evidence_item_id": item["id"],
                        "source": "annotated",
                    }
                )

        for link in data.get("attack", []):
            claim_text = claims.get(link.get("to"))
            evidence_text = evidences.get(link.get("from"))
            if claim_text and evidence_text:
                attack_pairs.append(
                    {
                        "claim_text": claim_text,
                        "evidence_text": evidence_text,
                        "label": "attack",
                        "claim_item_id": item["id"],
                        "evidence_item_id": item["id"],
                        "source": "annotated",
                    }
                )

    return support_pairs, attack_pairs


def neutral_pairs(
    items: list[dict[str, Any]],
    target_count: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Create cross-topic neutral pairs with low lexical overlap."""
    rng = random.Random(seed)
    claims: list[tuple[int, str]] = []
    evidences: list[tuple[int, str]] = []
    for item in items:
        for claim in item["data"].get("claim", []):
            claims.append((item["id"], claim["text"]))
        for evidence in item["data"].get("Evidence", []):
            evidences.append((item["id"], evidence["text"]))

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = target_count * 200
    while len(pairs) < target_count and attempts < max_attempts:
        attempts += 1
        claim_item_id, claim_text = rng.choice(claims)
        evidence_item_id, evidence_text = rng.choice(evidences)
        key = (claim_text, evidence_text)
        if claim_item_id == evidence_item_id or key in seen:
            continue
        overlap = jaccard(claim_text, evidence_text)
        if overlap > 0.08:
            continue
        seen.add(key)
        pairs.append(
            {
                "claim_text": claim_text,
                "evidence_text": evidence_text,
                "label": "neutral",
                "claim_item_id": claim_item_id,
                "evidence_item_id": evidence_item_id,
                "source": "cross_topic_low_overlap",
                "lexical_overlap": round(overlap, 4),
            }
        )

    if len(pairs) < target_count:
        raise RuntimeError(
            f"Only generated {len(pairs)} neutral pairs; requested {target_count}."
        )
    return pairs


def main() -> None:
    """Generate corrected component and relation training files."""
    raw = load_json(SOURCE_DATASET)
    component_items = [item for item in (normalize_item(item) for item in raw) if item]
    support, attack = linked_pairs(component_items)
    target_neutral_count = max(len(support), len(attack))
    neutral = neutral_pairs(component_items, target_neutral_count)

    relation_items = [*support, *attack, *neutral]
    random.Random(SEED).shuffle(relation_items)

    dump_json(COMPONENT_DATASET, component_items)
    dump_json(RELATION_DATASET, relation_items)

    print(f"component_items={len(component_items)} -> {COMPONENT_DATASET}")
    print(f"relation_pairs={len(relation_items)} -> {RELATION_DATASET}")
    print("relation_label_counts=", dict(Counter(pair["label"] for pair in relation_items)))


if __name__ == "__main__":
    main()
