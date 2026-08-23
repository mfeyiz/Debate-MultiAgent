#!/usr/bin/env python3
"""Faz 4: build the locked gold holdout (eval_holdout_scenarios_v2.json).

Trustworthy "real performance" measurement requires an evaluation set that the
training pipeline never touched and that comes from an INDEPENDENT generator
family. This script:

1. Generates paragraphs with HOLDOUT_GENERATOR_MODEL (a family used by neither the
   training generators nor the judges), on topics NOT seen in training.
2. Labels each with the 2 consensus judges.
3. Keeps ONLY paragraphs where the judges are UNANIMOUS on every component and every
   relation (full-paragraph consensus) and the graph schema holds. This is the
   automated stand-in for human verification: ambiguous cases are discarded rather
   than guessed, so the surviving gold is high-confidence.
4. Writes scenario format consumed by scripts/evaluate_real_scenarios.py and LOCKS it.

This file must never be used for training or hyper-parameter tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

import httpx

import generate_dataset_v5 as gen
import consensus_label as cons
from argmine_common import (
    HOLDOUT_GENERATOR_MODEL,
    JUDGE_MODELS,
    OPENROUTER_API_KEY,
    build_topic_pool,
    call_openrouter,
    normalized_text,
    normalized_topic,
    parse_json_loose,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "eval_holdout_scenarios_v2.json"
# Exclude ALL training topics (labeled file holds the full merged training set).
TRAIN_RAW = ROOT / "Data-Debate" / "gold" / "data" / "v5" / "labeled_paragraphs.jsonl"

# Curated unseen topics (different from training base topics) + generic unseen pool.
HOLDOUT_TOPICS = [
    "Beyin-bilgisayar arayüzlerinin engelli bireylerin yaşamına etkisi",
    "Yapay zeka ile psikolojik danışmanlık hizmetlerinin sınırları",
    "Sanal gerçeklik sınıflarının öğrenme kalitesine etkisi",
    "Akıllı şehir sensörlerinin mahremiyet ve trafik yönetimi dengesi",
    "Karbon yakalama teknolojilerine kamu teşviki verilmesi",
    "Mikroplastik yasaklarının sanayi ve tüketici davranışına etkisi",
    "Deniz üstü rüzgar enerjisi yatırımlarının ekosistem üzerindeki etkisi",
    "Yaşlı bakımında robot asistanların kullanılması",
    "Kişisel sağlık verilerinin araştırma amacıyla anonimleştirilerek paylaşılması",
    "Gençlere yönelik zorunlu finansal okuryazarlık eğitimi",
    "Banka kredi skorlamasında algoritmik şeffaflık zorunluluğu",
    "Şirketlerin çalışan e-postalarını yapay zeka ile denetlemesi",
    "Dijital sanat eserlerinin telif ve sahiplik tartışması",
    "Spor müsabakalarında yarı otomatik hakem sistemlerinin güvenilirliği",
    "Kültür mirası alanlarında ziyaretçi kapasitesi sınırı",
    "Afet sonrası geçici konutların kalıcı mahallelere dönüşmesi",
]


def training_topic_keys() -> set[str]:
    return {normalized_topic(r["topic"]) for r in read_jsonl(TRAIN_RAW)}


_UNBOUNDED = asyncio.Semaphore(10_000)  # inner calls; concurrency is bounded by the worker pool


async def generate_candidate(client: httpx.AsyncClient, topic: str, model: str) -> dict | None:
    rng = random.Random(abs(hash(("holdout", topic))) % (2**32))
    style = rng.choice(gen.WRITING_STYLES)
    shape = rng.choice(gen.SHAPE_PROFILES)
    ids = rng.choice(gen.ID_SETS)
    content = await call_openrouter(
        client, model, gen.SYSTEM_PROMPT,
        gen.build_user_prompt(topic, style, shape, ids),
        temperature=0.85, force_json=True, label=f"holdout:{topic[:25]}",
    )
    if not content:
        return None
    try:
        parsed = parse_json_loose(content)
    except Exception:  # noqa: BLE001
        return None
    item = parsed[0] if isinstance(parsed, list) and parsed else parsed
    cleaned, _ = gen.validate_paragraph(item, topic)
    return cleaned


def full_consensus_scenario(paragraph: dict, judge_outputs: list) -> dict | None:
    """Build a gold scenario from per-item UNANIMOUS judge labels.

    Each kept component/relation has both judges agreeing (high-confidence gold);
    items where judges disagree are dropped rather than guessed. The paragraph is
    kept only if a valid argument scenario survives (>=3 components, >=2 relations,
    >=1 argumentative, >=1 claim). This mirrors the training-time consensus rule.
    """
    comp_maps = [j[0] for j in judge_outputs]
    rel_maps = [j[1] for j in judge_outputs]

    comp_final: dict[str, str] = {}
    components_out = []
    for c in paragraph["components"]:
        votes = [m.get(c["id"]) for m in comp_maps]
        label, status = cons.majority_label(votes)
        if status != "agree":
            continue  # drop only the ambiguous component, keep the paragraph
        comp_final[c["id"]] = label
        components_out.append({"id": c["id"], "label": label, "text": c["text"]})

    relations_out = []
    for r in paragraph["relations"]:
        if r["from"] not in comp_final or r["to"] not in comp_final:
            continue  # endpoint had no consensus
        votes = [m.get((r["from"], r["to"])) for m in rel_maps]
        label, status = cons.majority_label(votes)
        if status != "agree":
            continue
        action, final_label = cons.apply_relation_schema(
            label, comp_final[r["from"]], comp_final[r["to"]]
        )
        if action == "drop" or not final_label:
            continue
        relations_out.append({"from": r["from"], "to": r["to"], "label": final_label})

    labels = Counter(c["label"] for c in components_out)
    has_arg = any(rel["label"] in {"support", "attack"} for rel in relations_out)
    if len(components_out) < 3 or len(relations_out) < 2 or not has_arg:
        return None
    if labels["claim"] < 1:
        return None
    return {
        "data": {
            "link": "",
            "text": paragraph["text"],
            "type": "tartışma",
            "topic": paragraph["topic"],
            "components": components_out,
            "relations": relations_out,
        }
    }


async def async_main(args: argparse.Namespace) -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("HATA: OPENROUTER_API_KEY yok.")
    if len(JUDGE_MODELS) < 2:
        raise SystemExit("HATA: en az 2 JUDGE_MODELS gerekli.")

    seen_train = training_topic_keys()
    pool = [t for t in build_topic_pool(args.target * 5) if normalized_topic(t) not in seen_train]
    random.Random(7).shuffle(pool)
    topics = HOLDOUT_TOPICS + pool  # curated unseen first
    print(
        f"Gold holdout. generator={HOLDOUT_GENERATOR_MODEL}; judges={JUDGE_MODELS}; "
        f"hedef={args.target}; aday konu havuzu={len(topics)}; per-call timeout={args.stage_timeout}s",
        flush=True,
    )

    scenarios: list[dict] = []
    seen_text: set[str] = set()
    stats = Counter()
    lock = asyncio.Lock()

    async def produce_one(client: httpx.AsyncClient, topic: str) -> None:
        """Generate + judge one paragraph with hard per-stage timeouts.

        A hung provider call is abandoned (skipped) instead of stalling the run.
        """
        try:
            cand = await asyncio.wait_for(
                generate_candidate(client, topic, HOLDOUT_GENERATOR_MODEL), timeout=args.stage_timeout
            )
            if not cand:
                return
            judged = await asyncio.wait_for(
                cons.judge_paragraph(client, cand, JUDGE_MODELS, _UNBOUNDED), timeout=args.stage_timeout
            )
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - skip slow/broken candidates
            async with lock:
                stats["timeout_or_error"] += 1
            return
        async with lock:
            stats["generated"] += 1
            scenario = full_consensus_scenario(cand, judged)
            if scenario is None:
                stats["rejected_no_consensus"] += 1
                return
            tkey = normalized_text(scenario["data"]["text"])
            if tkey in seen_text:
                stats["dup"] += 1
                return
            seen_text.add(tkey)
            scenario["id"] = len(scenarios) + 1
            scenarios.append(scenario)

    async with httpx.AsyncClient(timeout=90.0) as client:
        idx = 0
        while len(scenarios) < args.target and idx < len(topics):
            batch = topics[idx : idx + args.concurrency]
            idx += args.concurrency
            await asyncio.gather(*[produce_one(client, t) for t in batch])
            print(
                f"  kabul {len(scenarios)}/{args.target} | üretilen {stats['generated']} | "
                f"red {stats['rejected_no_consensus']} | timeout/hata {stats['timeout_or_error']} "
                f"| ilerleme {idx}/{len(topics)} konu",
                flush=True,
            )

    scenarios = scenarios[: args.target]
    OUT_PATH.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comp = Counter(c["label"] for s in scenarios for c in s["data"]["components"])
    rel = Counter(r["label"] for s in scenarios for r in s["data"]["relations"])
    print(f"\nKilitlendi: {OUT_PATH} ({len(scenarios)} senaryo)", flush=True)
    print(f"Component etiketleri: {dict(comp)}", flush=True)
    print(f"Relation etiketleri:  {dict(rel)}", flush=True)
    print(f"İstatistik: {dict(stats)}", flush=True)
    if len(scenarios) < args.target:
        print(f"UYARI: hedefe ulaşılamadı ({len(scenarios)}/{args.target}).", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--stage-timeout", type=float, default=75.0,
                        help="Hard timeout (s) per generate/judge stage; hung calls are skipped.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
