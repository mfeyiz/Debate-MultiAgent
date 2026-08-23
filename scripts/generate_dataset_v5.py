#!/usr/bin/env python3
"""Faz 1: generate raw argument-mining paragraphs for the v5 dataset.

Unlike the legacy generator, the model's own component/relation labels are kept
ONLY as hints. The authoritative labels are decided later by independent judges
in consensus_label.py. This breaks the single-generator label monoculture that
capped the previous models.

Output: Data-Debate/gold/data/v5/raw_paragraphs.jsonl
Each line: {id, topic, generator, text, components:[{id,text,gen_label}],
            relations:[{from,to,gen_label}]}
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from argmine_common import (
    GENERATOR_MODELS,
    OPENROUTER_API_KEY,
    build_topic_pool,
    call_openrouter,
    normalized_text,
    normalized_topic,
    parse_json_loose,
    read_jsonl,
    text_contains,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "Data-Debate" / "gold" / "data" / "v5" / "raw_paragraphs.jsonl"

WRITING_STYLES = [
    "haber analizi", "forum yorumu", "akademik özet", "politika notu",
    "sosyal medya tartışması", "belediye meclisi tutanağı", "uzman görüşü",
    "rapor değerlendirmesi", "köşe yazısı", "panel konuşması özeti",
]

# We deliberately ask for a spread of argumentative configurations so the judges
# later see balanced claim/evidence/other and support/attack/none material.
SHAPE_PROFILES = [
    "Net bir destekleyici kanıt (support) ve bir karşı/sınırlayıcı iddia (attack) bulunsun.",
    "Konuya benzeyen ama farklı dönem/metrik/varlık içeren, iddiayı ne destekleyen ne çürüten "
    "ilişkisiz bir cümle (none adayı) mutlaka olsun.",
    "Sayısal veri içeren bir kanıt ile bu veriyle ÇELİŞEN bir karşı iddia (attack) bulunsun.",
    "Bir koşula bağlı iddia ve bu koşul sağlanmadığında geçerliliği değişen bir kanıt bulunsun.",
    # Hard contrastive claim/evidence pairs (the boundary the model struggles with):
    "Aynı paragrafta YAN YANA şunlar olsun: (a) kaynak/sayı içeren gerçek bir kanıt "
    "(örn. 'X raporuna göre oran %20 arttı'), ve (b) kulağa olgusal gelen ama kaynaksız bir "
    "değerlendirme iddiası (örn. 'Bu uygulama maliyetleri azaltır'). İkisi karışmasın.",
    # Minority-class balance: more evidence + an other context sentence.
    "En az 2 farklı somut kanıt (biri sayısal/kaynaklı, biri gözlem/örnek) ve 1 argüman dışı "
    "bağlam cümlesi (yöntem, kapsam veya tanım) içersin.",
]

SYSTEM_PROMPT = """\
Sen Türkçe argüman madenciliği için veri üreten uzman bir dil mühendisisin.
Görevin: verilen tartışma konusunda DOĞAL, akıcı, gerçekçi bir Türkçe paragraf yazmak ve
paragraftaki cümleleri argüman bileşenlerine bölmek.

Çıktı YALNIZCA şu JSON nesnesi olsun (markdown, açıklama yok):
{
  "text": "tüm paragraf (5-8 cümle, ~90-150 kelime)",
  "components": [
    {"id": "C1", "text": "paragraftan HARFİ HARFİNE alınmış bir cümle", "hint": "claim"},
    {"id": "E1", "text": "...", "hint": "evidence"},
    {"id": "O1", "text": "...", "hint": "other"}
  ],
  "relations": [
    {"from": "E1", "to": "C1", "hint": "support"},
    {"from": "C2", "to": "C1", "hint": "attack"},
    {"from": "O1", "to": "C1", "hint": "none"}
  ]
}

Kurallar:
1. components[].text değerleri paragraf "text" içinde birebir geçmeli (kopyala-yapıştır).
2. Paragrafı anlamlı cümle/önerme birimlerine böl; her cümle bir component olsun (4-6 component).
3. "hint" senin ön tahminin; sadece claim/evidence/other kullan. Bu etiket bağlayıcı değildir,
   sonradan bağımsız olarak yeniden değerlendirilecektir; yine de elinden gelen en doğru tahmini ver.
4. relations[].from ve to yalnızca components id'lerine referans versin; hint sadece support/attack/none.
5. En az 3 component ve en az 2 relation üret; mümkünse support, attack ve none çeşitliliği olsun.
6. Şablon cümleler, "Bu başlık..." kalıbı ve yapay liste dili KULLANMA; gerçek metin gibi yaz.
7. claim = savunulabilir iddia/sonuç/risk/fayda/öneri/değerlendirme. evidence = somut veri, araştırma,
   örnek, oran, gözlem, kaynaklı bulgu. other = argüman dışı bağlam (tanım, yöntem, kapsam, sayfa bilgisi).
"""


def build_user_prompt(topic: str, style: str, shape: str, ids: tuple[str, ...]) -> str:
    c1, e1, c2, o1 = ids
    return (
        f'Tartışma konusu: "{topic}"\n\n'
        f"Üslup: {style}. {shape}\n"
        f"Component id'leri için şu deseni kullan: {c1} (ana iddia), {e1} (kanıt), "
        f"{c2} (karşı/ikincil iddia), {o1} (argüman dışı bağlam). Gerekirse ek id ekleyebilirsin.\n"
        f"{o1} cümlesi risk/fayda/sonuç/oran/veri içermesin; yalnız kapsam, tanım, yöntem veya "
        "sayfa bilgisi olsun.\n"
        "Yalnızca tek bir JSON nesnesi döndür."
    )


ID_SETS = [
    ("C1", "E1", "C2", "O1"),
    ("K1", "D1", "K2", "B1"),
    ("A1", "V1", "A2", "M1"),
]


def validate_paragraph(raw: Any, topic: str) -> tuple[dict | None, list[str]]:
    """Validate a generated paragraph; labels are kept as hints only."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, ["not an object"]
    text = str(raw.get("text", "")).strip()
    if len(text) < 200 or len(text.split()) < 45:
        errors.append("paragraph too short")

    raw_components = raw.get("components", [])
    raw_relations = raw.get("relations", [])
    if not isinstance(raw_components, list) or not isinstance(raw_relations, list):
        return None, ["components/relations must be lists"]

    components: list[dict] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for comp in raw_components:
        if not isinstance(comp, dict):
            continue
        cid = str(comp.get("id", "")).strip()
        ctext = str(comp.get("text", "")).strip()
        hint = str(comp.get("hint", comp.get("label", ""))).strip().lower()
        key = normalized_text(ctext)
        if not cid or cid in seen_ids:
            continue
        if len(ctext) < 12 or not text_contains(text, ctext) or key in seen_text:
            continue
        seen_ids.add(cid)
        seen_text.add(key)
        components.append({"id": cid, "text": ctext, "gen_label": hint})

    if len(components) < 3:
        errors.append("needs at least 3 valid components")

    valid_ids = {c["id"] for c in components}
    relations: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue
        src = str(rel.get("from", "")).strip()
        tgt = str(rel.get("to", "")).strip()
        hint = str(rel.get("hint", rel.get("label", ""))).strip().lower()
        if src not in valid_ids or tgt not in valid_ids or src == tgt:
            continue
        if (src, tgt) in seen_pairs:
            continue
        seen_pairs.add((src, tgt))
        relations.append({"from": src, "to": tgt, "gen_label": hint})

    if len(relations) < 2:
        errors.append("needs at least 2 valid relations")

    if errors:
        return None, errors
    return {"topic": topic, "text": text, "components": components, "relations": relations}, []


async def process_topic(
    client: httpx.AsyncClient,
    topic: str,
    model: str,
    sem: asyncio.Semaphore,
    state: dict,
    lock: asyncio.Lock,
    target: int,
) -> None:
    async with sem:
        async with lock:
            if len(state["rows"]) >= target:
                return
        rng = random.Random(abs(hash(topic)) % (2**32))
        style = rng.choice(WRITING_STYLES)
        shape = rng.choice(SHAPE_PROFILES)
        ids = rng.choice(ID_SETS)
        content = await call_openrouter(
            client,
            model,
            SYSTEM_PROMPT,
            build_user_prompt(topic, style, shape, ids),
            temperature=0.9,
            force_json=True,
            label=f"gen:{topic[:30]}",
        )
        if not content:
            return
        try:
            parsed = parse_json_loose(content)
        except Exception as exc:  # noqa: BLE001
            print(f"   [parse-fail] {topic[:40]}: {exc}", flush=True)
            return
        items = parsed if isinstance(parsed, list) else [parsed]
        async with lock:
            kept = 0
            rejected: Counter[str] = Counter()
            for item in items:
                cleaned, errs = validate_paragraph(item, topic)
                if cleaned is None:
                    rejected.update(errs)
                    continue
                tkey = normalized_text(cleaned["text"])
                if tkey in state["texts"]:
                    rejected["duplicate paragraph"] += 1
                    continue
                cleaned["id"] = len(state["rows"]) + 1
                cleaned["generator"] = model
                state["rows"].append(cleaned)
                state["texts"].add(tkey)
                kept += 1
                if len(state["rows"]) >= target:
                    break
            if kept:
                write_jsonl(OUT_PATH, state["rows"])
            msg = f"   [{model.split('/')[-1]}] +{kept} -> total {len(state['rows'])}"
            if rejected:
                msg += f" (ret: {dict(rejected.most_common(3))})"
            print(msg, flush=True)


async def async_main(args: argparse.Namespace) -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("HATA: .env içinde OPENROUTER_API_KEY yok.")
    if not GENERATOR_MODELS:
        raise SystemExit("HATA: GENERATOR_MODELS boş.")

    existing = []
    if OUT_PATH.exists() and not args.fresh:
        existing = read_jsonl(OUT_PATH)
    state = {
        "rows": existing,
        "texts": {normalized_text(r["text"]) for r in existing},
    }
    print(
        f"v5 üretim. generators={GENERATOR_MODELS}; mevcut={len(existing)}; hedef={args.target}",
        flush=True,
    )
    if len(state["rows"]) >= args.target:
        print("Hedef zaten tamam.", flush=True)
        return

    exclude: set[str] = set()
    if args.exclude_topics_from and args.exclude_topics_from.exists():
        exclude = {normalized_topic(r.get("topic", "")) for r in read_jsonl(args.exclude_topics_from)}
        print(f"Hariç tutulan konu sayısı: {len(exclude)}", flush=True)
    pool = [t for t in build_topic_pool(args.target * 6) if normalized_topic(t) not in exclude]
    random.Random(args.topic_seed).shuffle(pool)
    print(f"Kullanılabilir yeni konu: {len(pool)}", flush=True)
    # Assign each topic a generator, round-robin across the model list.
    assignments = [(pool[i], GENERATOR_MODELS[i % len(GENERATOR_MODELS)]) for i in range(len(pool))]

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=200.0) as client:
        tasks = [
            process_topic(client, topic, model, sem, state, lock, args.target)
            for topic, model in assignments
        ]
        await asyncio.gather(*tasks)

    rows = state["rows"][: args.target]
    write_jsonl(OUT_PATH, rows)
    by_gen = Counter(r["generator"] for r in rows)
    print(f"\nTamamlandı: {len(rows)} paragraf -> {OUT_PATH}", flush=True)
    print(f"Generator dağılımı: {dict(by_gen)}", flush=True)
    print(
        f"Toplam component adayı: {sum(len(r['components']) for r in rows)}; "
        f"relation adayı: {sum(len(r['relations']) for r in rows)}",
        flush=True,
    )
    if len(rows) < args.target:
        print(
            f"UYARI: hedefe ulaşılamadı ({len(rows)}/{args.target}); scripti tekrar çalıştır.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing output and restart.")
    parser.add_argument("--exclude-topics-from", type=Path, default=None,
                        help="JSONL with a 'topic' field; those topics are skipped (for augmenting).")
    parser.add_argument("--topic-seed", type=int, default=44, help="Shuffle seed for topic order.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
