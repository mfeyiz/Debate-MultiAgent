#!/usr/bin/env python3
"""Generate unseen holdout data and evaluate ModernBERT with strict macro-F1 gates."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.bert_service import ModernBERTPipeline  # noqa: E402


load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

OUTPUT_JSON = ROOT / "eval_holdout_scenarios.json"
COMPONENT_LABELS = ["claim", "evidence", "other"]
RELATION_LABELS = ["support", "attack", "none"]

HOLDOUT_TOPICS = [
    "Beyin-bilgisayar arayüzlerinin engelli bireylerin yaşamına etkisi",
    "Yapay zeka ile psikolojik danışmanlık hizmetlerinin sınırları",
    "Sanal gerçeklik sınıflarının öğrenme kalitesine etkisi",
    "Çocukların dijital oyun sürelerine yasal sınır getirilmesi",
    "Akıllı şehir sensörlerinin mahremiyet ve trafik yönetimi dengesi",
    "Kamu binalarında enerji verimliliği zorunluluklarının bütçeye etkisi",
    "Karbon yakalama teknolojilerine kamu teşviki verilmesi",
    "Kuraklık dönemlerinde tarımsal ürün deseninin devletçe yönlendirilmesi",
    "Mikroplastik yasaklarının sanayi ve tüketici davranışına etkisi",
    "Deniz üstü rüzgar enerjisi yatırımlarının ekosistem üzerindeki etkisi",
    "Yaşlı bakımında robot asistanların kullanılması",
    "Kişisel sağlık verilerinin araştırma amacıyla anonimleştirilerek paylaşılması",
    "Gençlere yönelik zorunlu finansal okuryazarlık eğitimi",
    "Kamu ihalelerinde yerli yazılım kullanım zorunluluğu",
    "Banka kredi skorlamasında algoritmik şeffaflık zorunluluğu",
    "Şirketlerin çalışan e-postalarını yapay zeka ile denetlemesi",
    "Dijital sanat eserlerinin telif ve sahiplik tartışması",
    "Spor müsabakalarında yarı otomatik hakem sistemlerinin güvenilirliği",
    "Kültür mirası alanlarında ziyaretçi kapasitesi sınırı",
    "Afet sonrası geçici konutların kalıcı mahallelere dönüşmesi",
]

SYSTEM_PROMPT = """\
Sen Türkçe argüman madenciliği (Argument Mining) alanında uzman bir test veri mühendisisin.
Görevin, eğitim verisinden farklı konularda doğal Türkçe paragraflar ve doğru argüman etiketleri üretmektir.

Her paragraf şu JSON şemasına uymalı:
{
  "id": <artışlı_tamsayı>,
  "data": {
    "link": "",
    "text": "tüm paragraf burada",
    "type": "",
    "topic": "konu başlığı",
    "components": [
      {"id": "C1", "label": "claim", "text": "bir iddia"},
      {"id": "E1", "label": "evidence", "text": "iddia için gerekçe, gözlem veya veri"},
      {"id": "C2", "label": "claim", "text": "karşı veya tamamlayıcı iddia"},
      {"id": "O1", "label": "other", "text": "sayfa bilgisi, yöntem notu, tanım veya argüman dışı bağlam"}
    ],
    "relations": [
      {"from": "E1", "to": "C1", "label": "support"},
      {"from": "C2", "to": "C1", "label": "attack"},
      {"from": "O1", "to": "C1", "label": "none"}
    ]
  }
}

Kurallar:
1. Component label değerleri sadece "claim", "evidence", "other" olabilir.
2. Relation label değerleri sadece "support", "attack", "none" olabilir.
3. "neutral", "non", "background" veya başka etiket kullanma.
4. claim = savunulabilir/itiraz edilebilir iddia, sonuç, risk, fayda, zarar, öneri veya değerlendirme cümlesi.
5. evidence = claim'i destekleyen veya çürüten somut gerekçe, örnek, veri, araştırma, gözlem, rapor veya kaynaklı bulgu.
6. other = argüman değildir; risk, fayda, sınırlılık, öneri, sonuç veya kanıt içermez. Sayfa bilgisi, yöntem notu, tanım, tarihçe, iletişim, tablo açıklaması gibi bağlam olmalı.
7. "BCI henüz yaygın değil", "maliyet yüksektir", "insan terapistin yerini alamaz" gibi cümleler other değil claim'dir.
8. "Araştırma gösterdi", "anket buldu", "örneğin", "%15 arttı" gibi cümleler other değil evidence'dır.
9. Her paragrafta en az 2 claim, en az 1 evidence, en az 1 other ve en az 2 relation olmalı.
10. none relation yalnızca gerçekten argüman dışı other component'ten bir claim'e kurulmalı.
11. Component text değerleri ana text içinde harfi harfine yer almalı.
12. Relation from/to değerleri yalnızca component id'lerine referans vermeli.
13. Etiketleri semantik olarak doğru ver; kalıp doldurmak için claim'i other veya attack'i none yapma.
14. Cevap sadece geçerli JSON dizisi olsun.
"""


def clean_response(text: str) -> str:
    """Strip optional markdown fences around a JSON response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def normalized_text(text: str) -> str:
    """Return a duplicate-detection key."""
    return re.sub(r"\s+", " ", text.strip().casefold())


def text_contains(haystack: str, needle: str) -> bool:
    """Case-insensitive containment with whitespace normalization."""
    return normalized_text(needle) in normalized_text(haystack)


CLAIM_RE = re.compile(
    r"\b("
    r"artır|arttır|azalt|düşür|yükselt|güçlendir|zayıflat|kolaylaştır|zorlaştır|"
    r"destekle|engelle|sağla|yol aç|neden ol|risk|tehdit|fayda|zarar|"
    r"gerekir|gereklidir|gerektirir|olmalıdır|olmamalıdır|savunulabilir|"
    r"uygundur|uygun değildir|mümkün değildir|yerini alamaz|umut vadet|"
    r"önemlidir|hayati|etkilidir|verimlidir|maliyetlidir|yüksektir|düşüktür|"
    r"sınırlıdır|risklidir|başarılıdır|başarısızdır|adil değildir|şeffaf değildir|"
    r"güvenilir değildir|mahremiyet|kaliteyi|başarıyı|motivasyonu|iletişimi"
    r")",
    re.IGNORECASE,
)
EVIDENCE_RE = re.compile(
    r"\b("
    r"örneğin|çünkü|zira|araştırma|çalışma|anket|rapor|veri|istatistik|"
    r"deney|pilot|bulgu|gözlem|kaynak|tüik|tcmb|oecd|üniversite|"
    r"gösterdi|göstermiştir|buldu|belirtti|belirtildi|bildirdi|raporladı|"
    r"ölçüldü|tespit edildi|kanıtladı|sonuçları|oran|yüzde|%"
    r")",
    re.IGNORECASE,
)
OTHER_RE = re.compile(
    r"\b("
    r"ana sayfa|çerez|gizlilik|site haritası|başvuru formu|iletişim|telefon|"
    r"e-posta|üyelik|abonelik|kategori|arşiv|paylaşım düğmesi|kaynakça|"
    r"yöntem|metodoloji|kapsam|örneklem|katılımcı|takvim|gündem|"
    r"tablo açıklaması|şekil açıklaması|teknik ek|veri indirme|lisans|"
    r"tanım|terim|tarihçe|adres|logo|sponsor|oturum"
    r")",
    re.IGNORECASE,
)


def semantic_label_error(label: str, text: str) -> str | None:
    """Reject obvious semantic label drift before it reaches holdout gold data."""
    normalized = normalized_text(text)
    claim_like = bool(CLAIM_RE.search(normalized))
    evidence_like = bool(EVIDENCE_RE.search(normalized) or re.search(r"\b\d+(?:[,.]\d+)?\b", normalized))
    other_like = bool(OTHER_RE.search(normalized))

    if label == "claim":
        if evidence_like and not claim_like:
            return "claim looks like evidence"
        if other_like and not claim_like:
            return "claim looks like other"
        if not claim_like:
            return "claim lacks stance/evaluation signal"
    elif label == "evidence":
        if other_like and not evidence_like:
            return "evidence looks like other"
        if not evidence_like:
            return "evidence lacks data/reason/source signal"
    elif label == "other":
        if claim_like:
            return "other looks like claim"
        if evidence_like:
            return "other looks like evidence"
        if not other_like:
            return "other lacks non-argument context signal"
    return None


def validate_item(
    item: dict[str, Any],
    topic: str,
    existing_texts: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one generated holdout item."""
    errors: list[str] = []
    data = item.get("data")
    if not isinstance(data, dict):
        return None, ["missing data object"]

    text = str(data.get("text", "")).strip()
    if len(text) < 120 or len(text.split()) < 16:
        errors.append("paragraph is too short")
    text_key = normalized_text(text)
    if text_key in existing_texts:
        errors.append("duplicate paragraph")

    raw_components = data.get("components", [])
    raw_relations = data.get("relations", [])
    if not isinstance(raw_components, list) or not isinstance(raw_relations, list):
        return None, ["components/relations must be lists"]

    components: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for component in raw_components:
        component_id = str(component.get("id", "")).strip()
        label = str(component.get("label", "")).strip()
        component_text = str(component.get("text", "")).strip()
        if not component_id or component_id in seen_ids:
            errors.append("missing or duplicate component id")
            continue
        if label not in COMPONENT_LABELS:
            errors.append(f"bad component label: {label}")
            continue
        if len(component_text) < 12 or not text_contains(text, component_text):
            errors.append(f"component text not found: {component_id}")
            continue
        semantic_error = semantic_label_error(label, component_text)
        if semantic_error:
            errors.append(semantic_error)
            continue
        seen_ids.add(component_id)
        components.append({"id": component_id, "label": label, "text": component_text})

    component_counts = Counter(component["label"] for component in components)
    if component_counts["claim"] < 2:
        errors.append("needs at least 2 claims")
    if component_counts["evidence"] < 1:
        errors.append("needs at least 1 evidence")
    if component_counts["other"] < 1:
        errors.append("needs at least 1 other")

    valid_ids = {component["id"] for component in components}
    label_by_id = {component["id"]: component["label"] for component in components}
    relations: list[dict[str, str]] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in raw_relations:
        source = str(relation.get("from", "")).strip()
        target = str(relation.get("to", "")).strip()
        label = str(relation.get("label", "")).strip()
        key = (source, target, label)
        if label not in RELATION_LABELS:
            errors.append(f"bad relation label: {label}")
            continue
        if source not in valid_ids or target not in valid_ids or source == target:
            errors.append("relation references invalid component id")
            continue
        if label == "none" and label_by_id.get(source) != "other":
            errors.append("none relation source must be other")
            continue
        if label in {"support", "attack"} and label_by_id.get(target) != "claim":
            errors.append("argumentative relation target must be claim")
            continue
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append({"from": source, "to": target, "label": label})

    if len(relations) < 2:
        errors.append("needs at least 2 valid relations")
    if not any(relation["label"] in {"support", "attack"} for relation in relations):
        errors.append("needs at least one argumentative relation")

    if errors:
        return None, errors
    return (
        {
            "id": int(item.get("id", 0)) if str(item.get("id", "")).isdigit() else 0,
            "data": {
                "link": str(data.get("link", "")),
                "text": text,
                "type": str(data.get("type", "")),
                "topic": str(data.get("topic") or topic),
                "components": components,
                "relations": relations,
            },
        },
        [],
    )


async def generate_examples(
    client: httpx.AsyncClient,
    topic: str,
    start_id: int,
    examples_per_topic: int,
) -> list[dict[str, Any]]:
    """Generate raw holdout examples from OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://logos-debate.app",
        "X-Title": "Logos Holdout Generator",
    }
    user_prompt = (
        f'Konu: "{topic}"\n\n'
        f"{examples_per_topic} adet benzersiz holdout örneği üret. "
        f"id değerleri {start_id} ile {start_id + examples_per_topic - 1} arasında olsun. "
        "Sadece geçerli JSON dizisi döndür."
    )
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.82,
    }
    for attempt in range(4):
        try:
            response = await client.post(
                OPENROUTER_API_BASE,
                headers=headers,
                json=payload,
                timeout=180.0,
            )
            response.raise_for_status()
            parsed = json.loads(clean_response(response.json()["choices"][0]["message"]["content"]))
            if isinstance(parsed, dict) and "samples" in parsed:
                parsed = parsed["samples"]
            if isinstance(parsed, dict) and "data" in parsed:
                parsed = [parsed]
            if isinstance(parsed, list):
                return parsed
            raise ValueError("expected a JSON array")
        except Exception as exc:  # noqa: BLE001 - generation retry logging.
            print(f"Hata ({topic}, deneme {attempt + 1}): {exc}", flush=True)
            await asyncio.sleep(2**attempt + 2)
    return []


async def build_eval_dataset(target_count: int, examples_per_topic: int) -> list[dict[str, Any]]:
    """Generate a fresh holdout dataset from topics outside the training list."""
    if not OPENROUTER_API_KEY:
        raise SystemExit("HATA: .env dosyasında OPENROUTER_API_KEY tanımlanmalı.")

    print(f"OpenRouter ile {target_count} adet yeni holdout test örneği üretiliyor...", flush=True)
    dataset: list[dict[str, Any]] = []
    existing_texts: set[str] = set()
    async with httpx.AsyncClient(timeout=200.0) as client:
        for topic in HOLDOUT_TOPICS:
            if len(dataset) >= target_count:
                break
            print(f"-> Üretiliyor: {topic}", flush=True)
            samples = await generate_examples(client, topic, len(dataset) + 1, examples_per_topic)
            rejected: Counter[str] = Counter()
            for sample in samples:
                cleaned, errors = validate_item(sample, topic, existing_texts)
                if cleaned is None:
                    rejected.update(errors)
                    continue
                cleaned["id"] = len(dataset) + 1
                dataset.append(cleaned)
                existing_texts.add(normalized_text(cleaned["data"]["text"]))
                if len(dataset) >= target_count:
                    break
            print(
                f"   [Tamamlandı] Toplam={len(dataset)} Ret={sum(rejected.values())}",
                flush=True,
            )
            if rejected:
                print(f"   Ret nedenleri: {dict(rejected.most_common(4))}", flush=True)
            await asyncio.sleep(0.5)
    if len(dataset) < target_count:
        raise SystemExit(f"Holdout hedefi tamamlanamadı: {len(dataset)}/{target_count}")
    return dataset


def evaluate_models(dataset: list[dict[str, Any]], comp_dir: Path, rel_dir: Path) -> dict[str, Any]:
    """Evaluate component and relation classifiers on a holdout dataset."""
    pipeline = ModernBERTPipeline(comp_dir, rel_dir)
    comp_true: list[str] = []
    comp_pred: list[str] = []
    comp_mismatches: list[dict[str, Any]] = []
    rel_true: list[str] = []
    rel_pred: list[str] = []
    rel_mismatches: list[dict[str, Any]] = []

    for item in dataset:
        data = item.get("data", {})
        topic = data.get("topic", "")
        components = data.get("components", [])
        relations = data.get("relations", [])
        component_by_id = {component["id"]: component for component in components}

        for component in components:
            expected = component["label"]
            predicted, confidence = pipeline._classify_component_unit(component["text"])
            comp_true.append(expected)
            comp_pred.append(predicted)
            if predicted != expected:
                comp_mismatches.append(
                    {
                        "topic": topic,
                        "text": component["text"],
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": confidence,
                    }
                )

        for relation in relations:
            source = component_by_id.get(relation["from"])
            target = component_by_id.get(relation["to"])
            if not source or not target:
                continue
            expected = relation["label"]
            predicted, confidence, probabilities = pipeline.classify_relation(target["text"], source["text"])
            rel_true.append(expected)
            rel_pred.append(predicted)
            if predicted != expected:
                rel_mismatches.append(
                    {
                        "topic": topic,
                        "claim": target["text"],
                        "source": source["text"],
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": confidence,
                        "probabilities": probabilities,
                    }
                )

    component_macro_f1 = f1_score(comp_true, comp_pred, labels=COMPONENT_LABELS, average="macro", zero_division=0)
    relation_macro_f1 = f1_score(rel_true, rel_pred, labels=RELATION_LABELS, average="macro", zero_division=0)
    payload = {
        "dataset_size": len(dataset),
        "component_total": len(comp_true),
        "relation_total": len(rel_true),
        "component_macro_f1": round(float(component_macro_f1), 4),
        "relation_macro_f1": round(float(relation_macro_f1), 4),
        "component_report": classification_report(
            comp_true,
            comp_pred,
            labels=COMPONENT_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "relation_report": classification_report(
            rel_true,
            rel_pred,
            labels=RELATION_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "component_confusion_matrix": {
            "labels": COMPONENT_LABELS,
            "matrix": confusion_matrix(comp_true, comp_pred, labels=COMPONENT_LABELS).tolist(),
        },
        "relation_confusion_matrix": {
            "labels": RELATION_LABELS,
            "matrix": confusion_matrix(rel_true, rel_pred, labels=RELATION_LABELS).tolist(),
        },
        "component_mismatches": comp_mismatches[:40],
        "relation_mismatches": rel_mismatches[:40],
        "passed": bool(component_macro_f1 >= 0.80 and relation_macro_f1 >= 0.80),
    }
    return payload


def print_human_report(metrics: dict[str, Any]) -> None:
    """Print a compact human-readable report."""
    print("\n" + "=" * 58)
    print(" HOLDOUT DEĞERLENDİRME SONUCU")
    print("=" * 58)
    print(f"Örnek sayısı: {metrics['dataset_size']}")
    print(f"Component macro-F1: {metrics['component_macro_f1']:.4f}")
    print(f"Relation macro-F1:  {metrics['relation_macro_f1']:.4f}")
    print(f"Gate: {'GEÇTİ' if metrics['passed'] else 'KALDI'}")
    print("\nComponent confusion matrix:")
    print(np.array(metrics["component_confusion_matrix"]["matrix"]))
    print("\nRelation confusion matrix:")
    print(np.array(metrics["relation_confusion_matrix"]["matrix"]))
    if metrics["component_mismatches"]:
        print("\nİlk component hataları:")
        for mismatch in metrics["component_mismatches"][:5]:
            print(
                f"- {mismatch['expected']} -> {mismatch['predicted']} "
                f"({mismatch['confidence']:.3f}): {mismatch['text']}"
            )
    if metrics["relation_mismatches"]:
        print("\nİlk relation hataları:")
        for mismatch in metrics["relation_mismatches"][:5]:
            print(
                f"- {mismatch['expected']} -> {mismatch['predicted']} "
                f"({mismatch['confidence']:.3f}): {mismatch['source']} -> {mismatch['claim']}"
            )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-model-dir", type=Path, default=ROOT / "models" / "candidate" / "component_classifier" / "artifact")
    parser.add_argument("--relation-model-dir", type=Path, default=ROOT / "models" / "candidate" / "relation_classifier" / "artifact")
    parser.add_argument("--load-from-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--target-count", type=int, default=150)
    parser.add_argument("--examples-per-topic", type=int, default=8)
    parser.add_argument("--write-metrics", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enforce-thresholds", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.load_from_file and args.load_from_file.exists():
        dataset = json.loads(args.load_from_file.read_text(encoding="utf-8"))
        if not args.json:
            print(f"Holdout test verisi yüklendi: {args.load_from_file} ({len(dataset)} örnek)")
    else:
        dataset = asyncio.run(build_eval_dataset(args.target_count, args.examples_per_topic))
        args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Yeni holdout test verisi kaydedildi: {args.output}")

    metrics = evaluate_models(dataset, args.component_model_dir, args.relation_model_dir)
    if args.write_metrics:
        args.write_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.write_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print_human_report(metrics)
    if args.enforce_thresholds and not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
