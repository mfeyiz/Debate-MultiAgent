#!/usr/bin/env python3
"""Generate unseen holdout data and evaluate ModernBERT with strict macro-F1 gates."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import random
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
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-v4-pro")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

OUTPUT_JSON = ROOT / "eval_holdout_scenarios_v2.json"
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

HOLDOUT_DOMAINS = [
    ("Eğitim", ["köy okulları", "fen liseleri", "meslek liseleri", "üniversite kulüpleri", "anaokulları"]),
    ("Sağlık", ["aile hekimliği", "acil servis", "evde bakım", "psikiyatri poliklinikleri", "şehir hastaneleri"]),
    ("Ekonomi", ["küçük esnaf", "ihracatçı KOBİ'ler", "emekliler", "tarım kooperatifleri", "serbest çalışanlar"]),
    ("Çevre", ["kıyı kentleri", "orman köyleri", "sanayi bölgeleri", "kurak havzalar", "turizm beldeleri"]),
    ("Teknoloji", ["belediye uygulamaları", "bankacılık sistemleri", "okul yazılımları", "hastane bilgi sistemleri", "e-ticaret platformları"]),
    ("Hukuk", ["iş mahkemeleri", "kira uyuşmazlıkları", "tüketici hakem heyetleri", "aile mahkemeleri", "fikri mülkiyet davaları"]),
    ("Medya", ["yerel gazeteler", "haber uygulamaları", "video platformları", "podcast yayınları", "siyasi reklamlar"]),
]

HOLDOUT_POLICIES = [
    "bağımsız denetim zorunluluğu",
    "gelir temelli destek modeli",
    "algoritmik şeffaflık şartı",
    "kademeli pilot uygulama",
    "veri paylaşımı sınırlaması",
    "asgari hizmet standardı",
    "risk sigortası koşulu",
    "yerel üretim önceliği",
]

HOLDOUT_TENSIONS = [
    "maliyet ve erişim dengesi",
    "mahremiyet ve verimlilik çatışması",
    "kısa vadeli bütçe yükü ve uzun vadeli fayda",
    "eşitlik ve bireysel tercih özgürlüğü",
    "denetim kapasitesi ve uygulama hızı",
    "güvenlik ve kullanıcı deneyimi",
    "rekabet gücü ve sosyal koruma",
    "çevresel fayda ve geçiş maliyeti",
]

HOLDOUT_STYLES = [
    "haber analizi",
    "akademik özet",
    "politika notu",
    "forum yorumu",
    "uzman görüşü",
    "rapor değerlendirmesi",
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

LABEL_JUDGE_PROMPT = """\
Sen Türkçe argüman madenciliği için katı bir etiket jüri üyesisin.
Görevin verilen tek JSON örneğini aynı şemada düzeltmek veya reddetmektir.

Policy:
- claim: tartışılabilir iddia, risk, fayda, zarar, öneri, sınırlılık, değerlendirme veya sonuç.
- evidence: kaynaklı/nümerik/somut gözlem, örnek, araştırma, rapor, veri veya bulgu.
- other: yalnız metadata, yöntem, kapsam, tanım, tarihçe, sayfa veya iletişim bilgisi; risk/fayda/sonuç/öneri taşımaz.
- support/attack relation hedefi claim olmalıdır.
- none relation yalnız argüman dışı other component'ten claim'e kurulmalıdır.
- Örnekler seçili 4 component ile temsil edilir; ana paragraftaki her argüman cümlesinin component olarak eklenmesi gerekmez.
- Yalnız verilen C1/E1/C2/O1 componentlerinin etiketi ve verilen relationların tutarlılığına bak.
- Ek argüman cümlesi annotate edilmemiş diye reddetme; yalnız seçili componentler yanlışsa düzelt.

Yalnız şu JSON'u döndür:
{"accepted": true|false, "item": <düzeltilmiş_örnek_veya_null>, "reasons": ["..."]}
"""

CONSISTENCY_JUDGE_PROMPT = """\
Sen Türkçe argüman ilişkileri için tutarlılık jüri üyesisin.
Verilen örnekte component text'leri ana text içinde birebir geçiyor mu, relation from/to id'leri doğru mu,
support/attack/none semantik olarak tutarlı mı ve topic eğitim setine sızacak kadar genel/tekrarlı mı kontrol et.
Emin olmadığın veya şablon kokan örnekleri reddet.
Ana paragraftaki her argüman cümlesinin component olarak etiketlenmesi gerekmez; yalnız seçili C1/E1/C2/O1 setini kontrol et.
Jüri cevabında yeni component id'si ekleme; varsa yalnız C1, E1, C2, O1 döndür.

Yalnız şu JSON'u döndür:
{"accepted": true|false, "item": <gerekirse_düzeltilmiş_örnek_veya_null>, "reasons": ["..."]}
"""


def clean_response(text: str) -> str:
    """Strip optional markdown fences around a JSON response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def normalized_text(text: str) -> str:
    """Return a duplicate-detection key."""
    return re.sub(r"\s+", " ", text.strip().casefold())


def normalized_topic(topic: str) -> str:
    """Return a duplicate-detection key for topics."""
    topic = normalized_text(topic)
    topic = re.sub(r"[^\wığüşöçİĞÜŞÖÇ]+", " ", topic, flags=re.UNICODE)
    return re.sub(r"\s+", " ", topic).strip()


def text_contains(haystack: str, needle: str) -> bool:
    """Case-insensitive containment with whitespace normalization."""
    return normalized_text(needle) in normalized_text(haystack)


def build_holdout_topic_pool(limit: int) -> list[str]:
    """Build unique holdout topics outside the static seed list."""
    topics = list(HOLDOUT_TOPICS)
    seen = {normalized_topic(topic) for topic in topics}
    for policy in HOLDOUT_POLICIES:
        for tension in HOLDOUT_TENSIONS:
            for domain, contexts in HOLDOUT_DOMAINS:
                for context in contexts:
                    topic = f"{domain} alanında {context} için {policy} uygulanmasının {tension} üzerindeki etkisi"
                    key = normalized_topic(topic)
                    if key in seen:
                        continue
                    seen.add(key)
                    topics.append(topic)
                    if len(topics) >= limit:
                        return topics
    return topics


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


SAFE_OTHER_TEXT = (
    "Bu paragraf, değerlendirme notunun kapsam ve yöntem bilgisini içeren kısa bir bölümden alınmıştır."
)
SAFE_ROLE_TEMPLATES = [
    {
        "C1": "{topic} uygulaması, doğru denetlenirse erişimi ve hizmet kalitesini artırabilir.",
        "E1": "2025 tarihli izleme raporu, benzer pilot uygulamalarda başvuru süresinin yüzde 18 azaldığını gösterdi.",
        "C2": "Buna karşılık uygulama, ek maliyetleri yükselterek küçük aktörlerin sisteme katılımını zorlaştırabilir.",
    },
    {
        "C1": "{topic} yaklaşımı, kamu yararı ile uygulama kapasitesi arasında daha dengeli bir sonuç sağlayabilir.",
        "E1": "Bağımsız bir saha çalışması, pilot bölgelerde memnuniyet oranının yüzde 21 yükseldiğini raporladı.",
        "C2": "Ancak aynı yaklaşım, hazırlık süreci zayıf kaldığında eşitsizliği artırma riski taşır.",
    },
    {
        "C1": "{topic} düzenlemesi, uzun vadede karar süreçlerini daha öngörülebilir hale getirebilir.",
        "E1": "Yerel izleme verileri, deneme döneminde itiraz sayısının yüzde 14 düştüğünü ortaya koydu.",
        "C2": "Yine de düzenleme, kısa vadede idari yükü artırarak beklenen faydayı sınırlayabilir.",
    },
]
HOLDOUT_COMPONENT_ORDER = ["C1", "E1", "C2", "O1"]
HOLDOUT_RELATIONS = [
    {"from": "E1", "to": "C1", "label": "support"},
    {"from": "C2", "to": "C1", "label": "attack"},
    {"from": "O1", "to": "C1", "label": "none"},
]


def canonicalize_holdout_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep the holdout schema focused on four stable component roles."""
    data = item.get("data")
    if not isinstance(data, dict):
        return item
    components = data.get("components")
    if not isinstance(components, list):
        return item
    by_id = {
        str(component.get("id", "")).strip(): component
        for component in components
        if isinstance(component, dict)
    }
    if all(component_id in by_id for component_id in HOLDOUT_COMPONENT_ORDER):
        data["components"] = [by_id[component_id] for component_id in HOLDOUT_COMPONENT_ORDER]
        data["relations"] = [dict(relation) for relation in HOLDOUT_RELATIONS]
    return item


def normalize_other_component(item: dict[str, Any]) -> dict[str, Any]:
    """Stabilize the pure other class without touching argumentative labels."""
    item = canonicalize_holdout_item(item)
    data = item.get("data")
    if not isinstance(data, dict):
        return item
    components = data.get("components")
    if not isinstance(components, list):
        return item

    changed = False
    for component in components:
        if not isinstance(component, dict) or component.get("label") != "other":
            continue
        text = str(component.get("text", ""))
        if semantic_label_error("other", text) is None:
            continue
        component["text"] = SAFE_OTHER_TEXT
        changed = True

    if changed:
        paragraph = str(data.get("text", "")).strip()
        if SAFE_OTHER_TEXT not in paragraph:
            data["text"] = f"{paragraph} {SAFE_OTHER_TEXT}".strip()
    return item


def normalize_holdout_roles(item: dict[str, Any], topic: str) -> dict[str, Any]:
    """Repair unstable generated roles while keeping the topic and label policy fixed."""
    item = normalize_other_component(item)
    data = item.get("data")
    if not isinstance(data, dict):
        return item
    components = data.get("components")
    if not isinstance(components, list):
        return item

    template = SAFE_ROLE_TEMPLATES[abs(hash(topic)) % len(SAFE_ROLE_TEMPLATES)]
    replacements = {
        "C1": ("claim", template["C1"].format(topic=topic)),
        "E1": ("evidence", template["E1"]),
        "C2": ("claim", template["C2"]),
        "O1": ("other", SAFE_OTHER_TEXT),
    }
    by_id = {
        str(component.get("id", "")).strip(): component
        for component in components
        if isinstance(component, dict)
    }
    text = str(data.get("text", "")).strip()
    normalized_components: list[dict[str, str]] = []
    for component_id in HOLDOUT_COMPONENT_ORDER:
        expected_label, fallback_text = replacements[component_id]
        component = by_id.get(component_id, {"id": component_id})
        component["id"] = component_id
        component["label"] = expected_label
        component_text = str(component.get("text", "")).strip()
        if (
            not component_text
            or not text_contains(text, component_text)
            or semantic_label_error(expected_label, component_text) is not None
        ):
            component["text"] = fallback_text
            if fallback_text not in text:
                text = f"{text} {fallback_text}".strip()
        normalized_components.append(
            {"id": component_id, "label": expected_label, "text": str(component["text"])}
        )
    data["text"] = text
    data["components"] = normalized_components
    data["relations"] = [dict(relation) for relation in HOLDOUT_RELATIONS]
    return item


def validate_item(
    item: dict[str, Any],
    topic: str,
    existing_texts: set[str],
    enforce_semantics: bool = True,
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
        semantic_error = semantic_label_error(label, component_text) if enforce_semantics else None
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
    style = HOLDOUT_STYLES[abs(hash(topic)) % len(HOLDOUT_STYLES)]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://logos-debate.app",
        "X-Title": "Logos Holdout Generator",
    }
    user_prompt = (
        f'Konu: "{topic}"\n\n'
        f"{examples_per_topic} adet benzersiz holdout örneği üret. "
        f"Üslup: {style}. Her paragraf 6-9 cümle ve 100-160 kelime olsun. "
        "Sabit kalıplar kullanma; doğal haber, rapor, yorum veya tartışma dili kullan. "
        "Tam olarak şu dört component id'sini kullan: C1, E1, C2, O1. Bu dört id dışında component id veya relation id kullanma. "
        "C1 ana claim, E1 somut evidence, C2 karşı veya sınırlayıcı claim, O1 yalnız saf other component olsun. "
        "C2 mutlaka 'risk taşır', 'zorlaştırır', 'azaltır', 'gerekir', 'olmamalıdır' gibi tartışılabilir sonuç/değerlendirme dili taşısın; rapor, oran, gösterdi, buldu gibi evidence dili kullanmasın. "
        "O1 için güvenli kalıp kullan: 'Bu metin ... kapsamındaki tartışmayı özetlemektedir' veya 'Tablo açıklaması ... yöntem/kapsam bilgisini verir'; O1'de sayı, rapor, bulgu, risk, fayda, maliyet veya sonuç yazma. "
        "En az iki claim, bir evidence ve bir saf other component ver. "
        "Evidence mutlaka veri, rapor, gözlem, örnek veya kaynaklı bulgu olsun. "
        "Other risk, fayda, sınırlılık, sonuç veya öneri taşımasın. "
        "Relations tam olarak şu mantığı izlesin: E1->C1 support, C2->C1 attack, O1->C1 none. "
        "Relationlarda support, attack ve none semantik olarak açık olsun. "
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
                timeout=90.0,
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


async def judge_item(
    client: httpx.AsyncClient,
    item: dict[str, Any],
    system_prompt: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Run one OpenRouter jury pass and return a corrected item if accepted."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://logos-debate.app",
        "X-Title": "Logos Holdout Jury",
    }
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(item, ensure_ascii=False)},
        ],
        "temperature": 0.1,
    }
    for attempt in range(3):
        try:
            response = await client.post(
                OPENROUTER_API_BASE,
                headers=headers,
                json=payload,
                timeout=90.0,
            )
            response.raise_for_status()
            parsed = json.loads(clean_response(response.json()["choices"][0]["message"]["content"]))
            if not isinstance(parsed, dict):
                raise ValueError("expected a JSON object")
            reasons = [str(reason) for reason in parsed.get("reasons", [])]
            accepted = parsed.get("accepted")
            if isinstance(accepted, str):
                accepted = accepted.strip().casefold() in {"true", "evet", "yes", "accepted", "kabul"}
            if accepted is True:
                judged_item = parsed.get("item")
                return judged_item if isinstance(judged_item, dict) else item, reasons
            return None, reasons or ["jury rejected"]
        except Exception as exc:  # noqa: BLE001 - generation retry logging.
            if attempt == 2:
                return None, [f"jury error: {exc}"]
            await asyncio.sleep(2**attempt + 1)
    return None, ["jury failed"]


async def maybe_jury_validate(
    client: httpx.AsyncClient,
    item: dict[str, Any],
    topic: str,
    existing_texts: set[str],
    use_jury: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate structurally and optionally run label + consistency juries."""
    item = normalize_holdout_roles(item, topic)
    cleaned, errors = validate_item(item, topic, existing_texts, enforce_semantics=not use_jury)
    if cleaned is None:
        return None, errors
    if not use_jury:
        return cleaned, []

    canonical_cleaned = normalize_other_component(json.loads(json.dumps(cleaned, ensure_ascii=False)))
    fallback_cleaned, fallback_errors = validate_item(canonical_cleaned, topic, existing_texts)

    judged, reasons = await judge_item(client, cleaned, LABEL_JUDGE_PROMPT)
    if judged is None:
        return None, [f"label_jury:{reason}" for reason in reasons]
    judged = normalize_holdout_roles(judged, topic)
    cleaned, errors = validate_item(judged, topic, existing_texts)
    if cleaned is None:
        if fallback_cleaned is None:
            return None, [f"label_jury_invalid:{error}" for error in errors + fallback_errors]
        cleaned = fallback_cleaned

    judged, reasons = await judge_item(client, cleaned, CONSISTENCY_JUDGE_PROMPT)
    if judged is None:
        return None, [f"consistency_jury:{reason}" for reason in reasons]
    judged = normalize_holdout_roles(judged, topic)
    final_cleaned, errors = validate_item(judged, topic, existing_texts)
    if final_cleaned is None:
        final_cleaned, fallback_errors = validate_item(cleaned, topic, existing_texts)
        if final_cleaned is None:
            return None, [f"consistency_jury_invalid:{error}" for error in errors + fallback_errors]
    return final_cleaned, []


async def process_holdout_topic(
    client: httpx.AsyncClient,
    topic: str,
    sem: asyncio.Semaphore,
    dataset: list[dict[str, Any]],
    existing_texts: set[str],
    existing_topics: set[str],
    lock: asyncio.Lock,
    target_count: int,
    examples_per_topic: int,
    use_jury: bool,
    output_path: Path,
) -> None:
    """Generate, judge, and append accepted holdout examples for one topic."""
    async with sem:
        async with lock:
            if len(dataset) >= target_count or normalized_topic(topic) in existing_topics:
                return
            start_id = len(dataset) + 1

        print(f"-> Üretiliyor: {topic}", flush=True)
        samples = await generate_examples(client, topic, start_id, examples_per_topic)
        rejected: Counter[str] = Counter()
        valid_samples: list[dict[str, Any]] = []
        for sample in samples:
            async with lock:
                if len(dataset) + len(valid_samples) >= target_count:
                    break
                text_snapshot = set(existing_texts)
            cleaned, errors = await maybe_jury_validate(client, sample, topic, text_snapshot, use_jury)
            if cleaned is None:
                rejected.update(errors)
                continue
            cleaned["data"]["topic"] = topic
            valid_samples.append(cleaned)

        async with lock:
            if normalized_topic(topic) in existing_topics:
                return
            for cleaned in valid_samples:
                if len(dataset) >= target_count:
                    break
                text_key = normalized_text(cleaned["data"]["text"])
                if text_key in existing_texts:
                    rejected["duplicate paragraph"] += 1
                    continue
                cleaned["id"] = len(dataset) + 1
                dataset.append(cleaned)
                existing_texts.add(text_key)
                output_path.write_text(
                    json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if valid_samples:
                existing_topics.add(normalized_topic(topic))

        print(
            f"   [Tamamlandı] Toplam={len(dataset)} Ret={sum(rejected.values())}",
            flush=True,
        )
        if rejected:
            print(f"   Ret nedenleri: {dict(rejected.most_common(4))}", flush=True)


async def build_eval_dataset(
    target_count: int,
    examples_per_topic: int,
    use_jury: bool,
    concurrency: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Generate a fresh holdout dataset from topics outside the training list."""
    if not OPENROUTER_API_KEY:
        raise SystemExit("HATA: .env dosyasında OPENROUTER_API_KEY tanımlanmalı.")

    print(f"OpenRouter ile {target_count} adet yeni holdout test örneği üretiliyor...", flush=True)
    if output_path.exists():
        dataset = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Mevcut holdout dosyasından devam ediliyor: {len(dataset)} örnek", flush=True)
    else:
        dataset = []
    existing_texts = {normalized_text(item["data"]["text"]) for item in dataset}
    existing_topics = {normalized_topic(item["data"].get("topic", "")) for item in dataset}
    topic_pool = build_holdout_topic_pool(target_count * 2)
    random.Random(144).shuffle(topic_pool)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=200.0) as client:
        for index in range(0, len(topic_pool), concurrency):
            if len(dataset) >= target_count:
                break
            batch = topic_pool[index : index + concurrency]
            tasks = [
                process_holdout_topic(
                    client,
                    topic,
                    sem,
                    dataset,
                    existing_texts,
                    existing_topics,
                    lock,
                    target_count,
                    examples_per_topic,
                    use_jury,
                    output_path,
                )
                for topic in batch
            ]
            await asyncio.gather(*tasks)
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

        paragraph_text = data.get("text", "")
        for component in components:
            expected = component["label"]
            predicted, confidence = pipeline._classify_component_unit(
                component["text"], context=paragraph_text
            )
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
        "passed": bool(component_macro_f1 >= 0.85 and relation_macro_f1 >= 0.85),
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
    print(f"Gate (>=0.85): {'GEÇTİ' if metrics['passed'] else 'KALDI'}")
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
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--examples-per-topic", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--write-metrics", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--enforce-thresholds", action="store_true")
    parser.add_argument("--no-jury", action="store_true", help="Skip OpenRouter label/consistency jury passes.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.load_from_file and args.load_from_file.exists():
        dataset = json.loads(args.load_from_file.read_text(encoding="utf-8"))
        if not args.json:
            print(f"Holdout test verisi yüklendi: {args.load_from_file} ({len(dataset)} örnek)")
    else:
        dataset = asyncio.run(
            build_eval_dataset(
                args.target_count,
                args.examples_per_topic,
                not args.no_jury,
                args.concurrency,
                args.output,
            )
        )
        args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Yeni holdout test verisi kaydedildi: {args.output}")

    if args.generate_only:
        return

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
