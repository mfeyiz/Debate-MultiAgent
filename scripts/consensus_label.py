#!/usr/bin/env python3
"""Faz 2: consensus relabeling of v5 paragraphs.

Each component and candidate relation is labeled INDEPENDENTLY by N judge models
(default 2, from different families than the generators). Only examples where the
judges agree survive. This removes the single-generator label noise that capped
the previous models. The project graph schema is enforced after consensus.

Input:  Data-Debate/gold/data/v5/raw_paragraphs.jsonl
Output: Data-Debate/gold/data/v5/labeled_paragraphs.jsonl
        Data-Debate/gold/data/v5/label_audit.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import httpx

from argmine_common import (
    COMPONENT_LABELS,
    JUDGE_MODELS,
    OPENROUTER_API_KEY,
    RELATION_LABELS,
    call_openrouter,
    parse_json_loose,
    read_jsonl,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "Data-Debate" / "gold" / "data" / "v5" / "raw_paragraphs.jsonl"
DEFAULT_OUT = ROOT / "Data-Debate" / "gold" / "data" / "v5" / "labeled_paragraphs.jsonl"
DEFAULT_AUDIT = ROOT / "Data-Debate" / "gold" / "data" / "v5" / "label_audit.json"


JUDGE_SYSTEM = """\
Sen Türkçe argüman madenciliği için titiz bir etiketleyicisin. Sana bir paragraf, paragraftaki
cümleler (component) ve aday ilişkiler (relation) verilecek. Görevin her birini ETİKETLEMEK.

Component etiketleri (yalnız bunlar):
- claim: savunulabilir/itiraz edilebilir iddia, sonuç, risk, fayda, zarar, öneri veya değerlendirme.
  Örn: "Bu teknoloji maliyetleri artırır", "X yapılmalıdır", "Y riski taşır".
- evidence: somut veri, araştırma, anket, rapor, oran, örnek, gözlem veya kaynaklı bulgu.
  Örn: "TÜİK 2023'te oranın %12 olduğunu açıkladı", "Pilot çalışmada başarı %30 arttı".
- other: argüman değildir; iddia/kanıt içermez. Tanım, yöntem, kapsam, sayfa bilgisi, idari bağlam.

Relation etiketleri (yalnız bunlar), "from" cümlesinin "to" cümlesine göre rolü:
- support: from, to'yu destekler/güçlendirir veya ona kanıt sunar.
- attack: from, to ile çelişir/onu zayıflatır veya karşı çıkar.
- none: from, to ile argümantatif ilişki kurmaz (ilgisiz, yalnızca bağlam, farklı koşul/dönem/metrik).

Kararlarını paragrafın BAĞLAMINA göre ver. Yalnızca şu JSON nesnesini döndür (açıklama yok):
{"components": [{"id": "C1", "label": "claim"}], "relations": [{"from": "E1", "to": "C1", "label": "support"}]}
"""


def build_judge_prompt(paragraph: dict) -> str:
    lines = [f'PARAGRAF:\n"{paragraph["text"]}"\n', "CÜMLELER (component):"]
    for c in paragraph["components"]:
        lines.append(f'- {c["id"]}: "{c["text"]}"')
    cid_text = {c["id"]: c["text"] for c in paragraph["components"]}
    lines.append("\nADAY İLİŞKİLER (relation): her biri için from cümlesinin to cümlesine rolünü etiketle.")
    for r in paragraph["relations"]:
        ft = cid_text.get(r["from"], "")
        tt = cid_text.get(r["to"], "")
        lines.append(f'- from={r["from"]} ("{ft[:90]}")  ->  to={r["to"]} ("{tt[:90]}")')
    lines.append(
        "\nHer component için claim/evidence/other, her relation için support/attack/none ver. "
        "Yalnızca verilen id'leri kullan. Tek JSON nesnesi döndür."
    )
    return "\n".join(lines)


def normalize_comp_label(label: str) -> str:
    label = str(label or "").strip().lower()
    if label == "background":
        return "other"
    return label if label in COMPONENT_LABELS else ""


def normalize_rel_label(label: str) -> str:
    label = str(label or "").strip().lower()
    if label == "neutral":
        return "none"
    return label if label in RELATION_LABELS else ""


def majority_label(votes: list[str | None]) -> tuple[str | None, str]:
    """Majority consensus over judge votes.

    - Needs >=2 valid (non-None) votes, otherwise 'incomplete'.
    - Keeps the label only if it has a strict majority among present votes
      (3 judges -> needs >=2 agreeing; 2 present -> needs unanimity).
    Returns (label_or_None, status) where status in {agree, disagree, incomplete}.
    """
    present = [v for v in votes if v is not None]
    if len(present) < 2:
        return None, "incomplete"
    label, count = Counter(present).most_common(1)[0]
    if count > len(present) / 2:
        return label, "agree"
    return None, "disagree"


def parse_judge_reply(content: str) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Return {component_id: label}, {(from,to): label} from a judge reply."""
    comp_labels: dict[str, str] = {}
    rel_labels: dict[tuple[str, str], str] = {}
    try:
        parsed = parse_json_loose(content)
    except Exception:  # noqa: BLE001
        return comp_labels, rel_labels
    if isinstance(parsed, list):  # tolerate a bare list of components
        parsed = {"components": parsed, "relations": []}
    if not isinstance(parsed, dict):
        return comp_labels, rel_labels
    for c in parsed.get("components", []) or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", "")).strip()
        lab = normalize_comp_label(c.get("label", ""))
        if cid and lab:
            comp_labels[cid] = lab
    for r in parsed.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        src = str(r.get("from", "")).strip()
        tgt = str(r.get("to", "")).strip()
        lab = normalize_rel_label(r.get("label", ""))
        if src and tgt and lab:
            rel_labels[(src, tgt)] = lab
    return comp_labels, rel_labels


def apply_relation_schema(label: str, source_label: str, target_label: str) -> tuple[str, str | None]:
    """Enforce project graph schema. Returns (action, final_label).

    action in {keep, fix, drop}. Mirrors relation_quality_policy.py.
    """
    if source_label == "other" and label != "none":
        return "fix", "none"
    if label in {"support", "attack"} and target_label != "claim":
        return "drop", None
    if label == "none" and source_label != "other":
        return "drop", None
    return "keep", label


async def judge_paragraph(
    client: httpx.AsyncClient, paragraph: dict, models: list[str], sem: asyncio.Semaphore
) -> list[tuple[dict[str, str], dict[tuple[str, str], str]]]:
    async with sem:
        prompt = build_judge_prompt(paragraph)
        replies = await asyncio.gather(
            *[
                call_openrouter(
                    client, m, JUDGE_SYSTEM, prompt,
                    temperature=0.0, force_json=True, label=f"judge:{m.split('/')[-1]}",
                )
                for m in models
            ]
        )
    return [parse_judge_reply(r or "") for r in replies]


async def async_main(args: argparse.Namespace) -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("HATA: .env içinde OPENROUTER_API_KEY yok.")
    judges = JUDGE_MODELS
    if len(judges) < 2:
        raise SystemExit("HATA: en az 2 JUDGE_MODELS gerekli (consensus için).")

    paragraphs = read_jsonl(args.input)
    if not paragraphs:
        raise SystemExit(f"HATA: girdi boş: {args.input}")
    print(f"Consensus etiketleme. judges={judges}; paragraf={len(paragraphs)}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    audit = {
        "judges": judges,
        "paragraphs": len(paragraphs),
        "component_agreement": {"agree": 0, "disagree": 0, "incomplete": 0},
        "relation_agreement": {"agree": 0, "disagree": 0, "incomplete": 0},
        "component_kept_labels": Counter(),
        "relation_kept_labels": Counter(),
        "relation_schema": Counter(),
    }
    labeled_rows: list[dict] = []

    progress = {"done": 0}
    lock = asyncio.Lock()

    def handle(paragraph: dict, judge_outputs: list) -> None:
        """Apply majority consensus + schema to one judged paragraph (mutates audit)."""
        comp_maps = [j[0] for j in judge_outputs]
        rel_maps = [j[1] for j in judge_outputs]

        final_components: list[dict] = []
        comp_final_label: dict[str, str] = {}
        for c in paragraph["components"]:
            cid = c["id"]
            votes = [m.get(cid) for m in comp_maps]
            label, status = majority_label(votes)
            audit["component_agreement"][status] += 1
            if status != "agree":
                continue
            comp_final_label[cid] = label
            audit["component_kept_labels"][label] += 1
            final_components.append(
                {"id": cid, "text": c["text"], "label": label, "judge_labels": votes}
            )

        final_relations: list[dict] = []
        for r in paragraph["relations"]:
            key = (r["from"], r["to"])
            votes = [m.get(key) for m in rel_maps]
            label, status = majority_label(votes)
            audit["relation_agreement"][status] += 1
            if status != "agree":
                continue
            src_label = comp_final_label.get(r["from"])
            tgt_label = comp_final_label.get(r["to"])
            if not src_label or not tgt_label:
                audit["relation_schema"]["drop_endpoint_no_consensus"] += 1
                continue
            action, final_label = apply_relation_schema(label, src_label, tgt_label)
            audit["relation_schema"][action] += 1
            if action == "drop" or not final_label:
                continue
            audit["relation_kept_labels"][final_label] += 1
            final_relations.append(
                {
                    "from": r["from"],
                    "to": r["to"],
                    "label": final_label,
                    "judge_labels": votes,
                    "schema_action": action,
                }
            )

        if final_components:
            labeled_rows.append(
                {
                    "id": paragraph.get("id"),
                    "topic": paragraph["topic"],
                    "generator": paragraph.get("generator"),
                    "text": paragraph["text"],
                    "components": final_components,
                    "relations": final_relations,
                }
            )

    async def worker(client: httpx.AsyncClient, paragraph: dict) -> None:
        # Worker pool: sem (inside judge_paragraph) bounds concurrency; no batch
        # barrier, so fast paragraphs are not blocked by slow stragglers.
        try:
            judge_outputs = await asyncio.wait_for(
                judge_paragraph(client, paragraph, judges, sem), timeout=args.stage_timeout
            )
        except Exception:  # noqa: BLE001 - skip stalled/broken paragraphs
            judge_outputs = None
        async with lock:
            if judge_outputs is not None:
                handle(paragraph, judge_outputs)
            else:
                audit["component_agreement"]["incomplete"] += 1
            progress["done"] += 1
            if progress["done"] % 25 == 0 or progress["done"] == len(paragraphs):
                print(f"  islendi {progress['done']}/{len(paragraphs)} paragraf", flush=True)

    async with httpx.AsyncClient(timeout=90.0) as client:
        await asyncio.gather(*[worker(client, p) for p in paragraphs])

    write_jsonl(args.output, labeled_rows)

    ca = audit["component_agreement"]
    ra = audit["relation_agreement"]
    comp_total = ca["agree"] + ca["disagree"]
    rel_total = ra["agree"] + ra["disagree"]
    audit["component_agreement_rate"] = round(ca["agree"] / comp_total, 4) if comp_total else 0.0
    audit["relation_agreement_rate"] = round(ra["agree"] / rel_total, 4) if rel_total else 0.0
    audit["kept_paragraphs"] = len(labeled_rows)
    audit["kept_components"] = sum(len(r["components"]) for r in labeled_rows)
    audit["kept_relations"] = sum(len(r["relations"]) for r in labeled_rows)
    # Counters -> dicts for JSON
    for k in ("component_kept_labels", "relation_kept_labels", "relation_schema"):
        audit[k] = dict(audit[k])
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nKaydedildi: {args.output} ({len(labeled_rows)} paragraf)", flush=True)
    print(f"Component agreement: {audit['component_agreement_rate']}  ({ca})", flush=True)
    print(f"Relation agreement:  {audit['relation_agreement_rate']}  ({ra})", flush=True)
    print(f"Tutulan component etiketleri: {audit['component_kept_labels']}", flush=True)
    print(f"Tutulan relation etiketleri:  {audit['relation_kept_labels']}", flush=True)
    print(f"Relation schema aksiyonları:  {audit['relation_schema']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--stage-timeout", type=float, default=120.0,
                        help="Hard timeout (s) per paragraph's judge round; stalled ones are skipped.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
