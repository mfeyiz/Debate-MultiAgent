#!/usr/bin/env python3
"""Append OpenRouter-generated Turkish argument-mining data to final_veri_seti.json.

The canonical dataset schema is intentionally the same as the production model:

- component labels: claim / evidence / other
- relation labels: support / attack / none

The script resumes from the current file, validates each generated paragraph,
deduplicates by paragraph text, and rebuilds the v4 training JSONL files once at
the end.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-v4-flash")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

OUTPUT_JSON = ROOT / "final_veri_seti.json"
COMPONENT_LABELS = {"claim", "evidence", "other"}
RELATION_LABELS = {"support", "attack", "none"}

BASE_TOPICS = [
    "Uzaktan eğitimin yüz yüze eğitimle karşılaştırılması",
    "Yapay zeka destekli ödev araçlarının öğrenme kalitesine etkisi",
    "STEM eğitiminin zorunlu müfredata dahil edilmesi",
    "Özel dershane ve sınav odaklı eğitim kültürünün etkileri",
    "Okul öncesi eğitimin zorunlu hale getirilmesi",
    "Sosyal medyanın gençlerin ruh sağlığına etkisi",
    "Kişisel verilerin gizliliği ve dijital gözetim",
    "Otonom araçların trafik güvenliğine etkisi",
    "Kripto paraların geleneksel bankacılık sistemine etkisi",
    "Yapay zekanın iş gücü piyasasına ve istihdama etkisi",
    "Asgari ücret artışlarının istihdam üzerindeki etkisi",
    "Enflasyonla mücadelede faiz politikasının tahmin gücü ve etkinliği",
    "Türkiye'nin dış ticaret açığı ve cari denge sorunları",
    "Vergi reformunun gelir dağılımı adaletine etkisi",
    "Konut fiyatlarındaki artış ve emlak balonu riski",
    "Aşılama programlarının toplum bağışıklığına katkısı",
    "Obezite ile mücadelede devlet politikalarının etkinliği",
    "Ruh sağlığı hizmetlerine erişimdeki eşitsizlikler",
    "Tütün ve alkol düzenlemelerinin halk sağlığına etkisi",
    "Sağlık turizminin geliştirilmesi ve ekonomik katkısı",
    "Yenilenebilir enerji yatırımlarının ekonomik getirisi",
    "Karbon vergisi uygulamasının sanayi sektörüne etkisi",
    "Plastik atık kirliliğinin deniz ekosistemine etkisi",
    "Nükleer enerjinin temiz enerji kaynağı olarak değerlendirilmesi",
    "Elektrikli araçlara geçişin çevresel etkileri",
    "Göç politikalarının toplumsal entegrasyona etkisi",
    "Toplumsal cinsiyet eşitliğinin iş gücüne yansıması",
    "Basın özgürlüğünün demokratik toplum için önemi",
    "Yaşlanan nüfusun sosyal güvenlik sistemine etkisi",
    "Hayvan hakları ve hayvan deneylerinin etik boyutu",
    "Yüksek hızlı tren yatırımlarının bölgesel kalkınmaya etkisi",
    "Toplu taşıma sistemlerinin şehir içi trafik sorunlarına etkisi",
    "Bisiklet yolları ve mikromobilite altyapısının şehir planlamasındaki yeri",
    "Zorunlu askerlik hizmetinin kaldırılması tartışması",
    "Oy kullanma yaşının 16'ya düşürülmesi",
    "Yapay zeka üretimlerinde telif hakları sorunu",
    "E-sporun resmi spor dalı olarak tanınması",
    "Sporda doping kullanımının adil rekabete etkisi",
    "GDO'lu gıdaların insan sağlığına etkisi tartışması",
    "Organik tarımın geleneksel tarıma göre sürdürülebilirliği",
]

EXPANSION_TOPICS = [
    "Dört günlük çalışma haftasının üretkenlik ve ücretler üzerindeki etkisi",
    "Çocuklara akıllı telefon kullanım yaş sınırı getirilmesi",
    "Kamu kurumlarında yapay zeka destekli karar sistemlerinin kullanılması",
    "Üniversite kontenjanlarının iş gücü ihtiyacına göre planlanması",
    "Online sınavlarda kamera gözetiminin mahremiyetle ilişkisi",
    "Şehir merkezlerinde araç giriş ücretinin trafik ve esnaf üzerindeki etkisi",
    "Kira artış sınırlamalarının konut arzına etkisi",
    "Turizm bölgelerinde kısa dönem kiralamanın yerel halk üzerindeki etkisi",
    "Kentsel dönüşüm projelerinde yerinde dönüşüm zorunluluğu",
    "Deprem sigortasının zorunlu kapsamının genişletilmesi",
    "İklim göçünün şehir planlaması ve sosyal politika üzerindeki etkisi",
    "Denizlerde av yasaklarının balıkçılık ekonomisine etkisi",
    "Tarımda su kotası uygulamasının üretim ve sürdürülebilirlik dengesi",
    "Gıda israfını azaltmak için marketlere bağış zorunluluğu getirilmesi",
    "Okullarda ücretsiz kahvaltı programının öğrenme başarısına etkisi",
    "Tam gün okul modelinin aileler ve öğrenciler üzerindeki etkisi",
    "Meslek liselerinin teknoloji sektörüne ara eleman yetiştirme kapasitesi",
    "Üniversitelerde zorunlu stajın mezun istihdamına etkisi",
    "Dil öğreniminde yapay zeka sohbet araçlarının kullanılması",
    "Açık kaynak yazılımların kamu kurumlarında tercih edilmesi",
    "Siber güvenlik derslerinin lise müfredatına eklenmesi",
    "Çocukların sosyal medya hesapları için ebeveyn onayı zorunluluğu",
    "Dijital platformlarda yaş doğrulama sistemlerinin güvenliği",
    "Algoritmik öneri sistemlerinin haber tüketimi üzerindeki etkisi",
    "Sahte haberle mücadelede devlet denetiminin ifade özgürlüğüne etkisi",
    "Gazetecilikte yapay zeka ile üretilen içeriklerin etiketlenmesi",
    "Kamu yayıncılığında tarafsızlık denetimi",
    "Siyasi reklamların sosyal medyada mikro hedefleme ile sunulması",
    "Seçim kampanyalarında deepfake kullanımına cezai yaptırım",
    "Oy verme işlemlerinde elektronik sistemlere geçiş",
    "Mahkemelerde uzaktan duruşma uygulamasının adalete erişime etkisi",
    "Arabuluculuk uygulamalarının iş davalarındaki yükü azaltması",
    "Cezaevlerinde rehabilitasyon programlarına ayrılan bütçenin artırılması",
    "Uyuşturucu bağımlılığıyla mücadelede cezalandırma yerine tedavi yaklaşımı",
    "Hastanelerde randevu sistemlerinin yapay zeka ile optimize edilmesi",
    "Aile hekimliği sisteminde performans kriterlerinin etkisi",
    "Nadir hastalık ilaçlarının kamu tarafından karşılanması",
    "Genetik tarama testlerinin yaygınlaştırılmasının etik sınırları",
    "Spor kulüplerinde yabancı oyuncu sınırlamasının rekabete etkisi",
    "Kadın spor liglerine medya görünürlüğü kotası getirilmesi",
    "Sanat kurumlarına kamu desteğinin kültürel çeşitliliğe etkisi",
    "Müzelerde ücretsiz giriş günlerinin kültürel erişime katkısı",
    "Tarihi yapıların turizme açılmasının koruma dengesi",
    "Dil sadeleşmesi politikalarının kültürel miras üzerindeki etkisi",
    "Kamu çalışanları için uzaktan çalışma hakkının kalıcı hale getirilmesi",
    "Sendikasız çalışanlar için toplu pazarlık modelleri",
    "Gig ekonomisi çalışanlarının sosyal güvence kapsamına alınması",
    "Robot vergisinin otomasyon yatırımlarına etkisi",
    "Merkez bankası dijital parasının bankacılık sistemine etkisi",
    "Nakit kullanımının azaltılmasının kayıt dışı ekonomiyle ilişkisi",
    "Kripto varlık vergilendirmesinin yatırımcı davranışlarına etkisi",
    "Enerji depolama yatırımlarının yenilenebilir enerji entegrasyonuna etkisi",
    "Hidrojen enerjisinin ağır sanayide karbonsuzlaşmaya katkısı",
    "Nükleer füzyon araştırmalarına kamu bütçesi ayrılması",
    "Laboratuvarda üretilen etin çevresel ve ekonomik etkileri",
    "Dikey tarımın şehir gıda güvenliği için uygulanabilirliği",
    "Drone ile teslimatın şehir trafiği ve güvenlik üzerindeki etkisi",
    "Elektrikli scooter düzenlemelerinin yaya güvenliğine etkisi",
    "Uydu internet hizmetlerinin kırsal dijital eşitsizliği azaltması",
    "Uzay madenciliğinin uluslararası hukuk ve ekonomi üzerindeki etkisi",
    "İnsan embriyosunda genetik düzenleme sınırları",
    "Evrensel temel gelirin çalışma motivasyonu üzerindeki etkisi",
]

SYSTEM_PROMPT = """\
Sen Türkçe argüman madenciliği (Argument Mining) ve tartışma analizi alanında uzman bir veri mühendisisin.
Görevin, verilen tartışma konusu hakkında doğal, yapaylıktan uzak, yüksek kaliteli Türkçe paragraflar ve bunlara ait argüman yapılarını üretmektir.

Her paragraf kesinlikle şu JSON şemasına uygun olmalı:
{
  "id": <artışlı_tamsayı>,
  "data": {
    "link": "",
    "text": "tüm paragraf burada",
    "type": "",
    "topic": "konu başlığı",
    "components": [
      {"id": "C1", "label": "claim", "text": "bir iddia"},
      {"id": "E1", "label": "evidence", "text": "iddia için gerekçe, veri veya gözlem"},
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
3. "neutral", "non", "background", "irrelevant" gibi etiketleri asla kullanma.
4. claim = savunulabilir/itiraz edilebilir iddia, sonuç, risk, fayda, zarar, öneri veya değerlendirme cümlesi.
5. evidence = claim'i destekleyen veya çürüten somut gerekçe, örnek, veri, araştırma, gözlem, rapor veya kaynaklı bulgu.
6. other = argüman değildir; risk, fayda, sınırlılık, öneri, sonuç veya kanıt içermez. Sayfa bilgisi, yöntem notu, tanım, tarihçe, iletişim, tablo açıklaması gibi bağlam olmalı.
7. "BCI henüz yaygın değil", "maliyet yüksektir", "insan terapistin yerini alamaz" gibi cümleler other değil claim'dir.
8. "Araştırma gösterdi", "anket buldu", "örneğin", "%15 arttı" gibi cümleler other değil evidence'dır.
9. Her paragrafta en az 2 claim, en az 1 evidence, en az 1 other ve en az 2 relation olmalı.
10. Relation çeşitliliği üret: support, attack ve none örnekleri mümkün olduğunca dengeli dağılsın.
11. none relation yalnızca gerçekten argüman dışı other component'ten bir claim'e kurulmalı.
12. Her component text değeri ana text içinde harfi harfine yer almalı.
13. Relation from/to değerleri yalnızca components dizisindeki id'lere referans vermeli.
14. Paragraflar gerçek hayattaki haber, forum yorumu, politika tartışması, akademik özet veya sosyal medya tartışması gibi çeşitli üsluplarda yazılmalı.
15. Şablonlaşmış tekrarları ve aynı cümle kalıplarını kullanma.
16. Cevap sadece geçerli JSON dizisi olsun; açıklama, markdown veya kod bloğu yazma.
"""


def clean_response(text: str) -> str:
    """Strip optional markdown fences around a JSON response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def normalized_text(text: str) -> str:
    """Return a duplicate-detection key for paragraph text."""
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
    """Reject obvious semantic label drift before it reaches gold data."""
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


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load an existing JSON dataset, returning an empty list if it is absent."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def relation_counts(dataset: list[dict[str, Any]]) -> Counter[str]:
    """Count relation labels in a dataset."""
    counts: Counter[str] = Counter()
    for item in dataset:
        for relation in item.get("data", {}).get("relations", []):
            counts[relation.get("label", "")] += 1
    return counts


def component_counts(dataset: list[dict[str, Any]]) -> Counter[str]:
    """Count component labels in a dataset."""
    counts: Counter[str] = Counter()
    for item in dataset:
        for component in item.get("data", {}).get("components", []):
            counts[component.get("label", "")] += 1
    return counts


def validate_item(
    item: dict[str, Any],
    topic: str,
    existing_texts: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate and normalize one generated dataset item."""
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
    seen_component_ids: set[str] = set()
    for component in raw_components:
        component_id = str(component.get("id", "")).strip()
        label = str(component.get("label", "")).strip()
        component_text = str(component.get("text", "")).strip()
        if not component_id or component_id in seen_component_ids:
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
        seen_component_ids.add(component_id)
        components.append({"id": component_id, "label": label, "text": component_text})

    counts = Counter(component["label"] for component in components)
    if counts["claim"] < 2:
        errors.append("needs at least 2 claim components")
    if counts["evidence"] < 1:
        errors.append("needs at least 1 evidence component")
    if counts["other"] < 1:
        errors.append("needs at least 1 other component")

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

    cleaned = {
        "id": int(item.get("id", 0)) if str(item.get("id", "")).isdigit() else 0,
        "data": {
            "link": str(data.get("link", "")),
            "text": text,
            "type": str(data.get("type", "")),
            "topic": str(data.get("topic") or topic),
            "components": components,
            "relations": relations,
        },
    }
    return cleaned, []


async def call_openrouter_async(
    client: httpx.AsyncClient,
    topic: str,
    start_id: int,
    examples_per_topic: int,
) -> list[dict[str, Any]]:
    """Call OpenRouter and return raw JSON examples."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://logos-debate.app",
        "X-Title": "Logos Dataset Generator",
    }
    user_prompt = (
        f'Konu: "{topic}"\n\n'
        f"{examples_per_topic} adet benzersiz örnek üret. "
        f"id değerleri {start_id} ile {start_id + examples_per_topic - 1} arasında olsun. "
        "Sadece geçerli JSON dizisi döndür."
    )
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.85,
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
            content = clean_response(response.json()["choices"][0]["message"]["content"])
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "samples" in parsed:
                parsed = parsed["samples"]
            if isinstance(parsed, dict) and "data" in parsed:
                parsed = [parsed]
            if isinstance(parsed, list):
                return parsed
            raise ValueError("expected a JSON array")
        except Exception as exc:  # noqa: BLE001 - log retry context for generation scripts.
            print(f"      [Hata - {topic} - Deneme {attempt + 1}]: {exc}", flush=True)
            await asyncio.sleep(2**attempt + 2)
    return []


async def process_topic(
    client: httpx.AsyncClient,
    topic: str,
    sem: asyncio.Semaphore,
    dataset: list[dict[str, Any]],
    existing_texts: set[str],
    lock: asyncio.Lock,
    target_count: int,
    examples_per_topic: int,
) -> None:
    """Generate and append valid examples for one topic."""
    async with sem:
        async with lock:
            if len(dataset) >= target_count:
                return
            start_id = len(dataset) + 1

        print(f"-> Başlatıldı: {topic}", flush=True)
        raw_samples = await call_openrouter_async(client, topic, start_id, examples_per_topic)

        valid_samples: list[dict[str, Any]] = []
        rejected: Counter[str] = Counter()
        async with lock:
            for raw in raw_samples:
                cleaned, errors = validate_item(raw, topic, existing_texts)
                if cleaned is None:
                    rejected.update(errors)
                    continue
                cleaned["id"] = len(dataset) + len(valid_samples) + 1
                valid_samples.append(cleaned)
                existing_texts.add(normalized_text(cleaned["data"]["text"]))
                if len(dataset) + len(valid_samples) >= target_count:
                    break

            if valid_samples:
                dataset.extend(valid_samples)
                OUTPUT_JSON.write_text(
                    json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        print(
            f"   [OK] {len(valid_samples)} kabul, {sum(rejected.values())} ret. "
            f"Toplam: {len(dataset)}",
            flush=True,
        )
        if rejected:
            print(f"   Ret nedenleri: {dict(rejected.most_common(4))}", flush=True)


def build_training_files(skip_external: bool) -> None:
    """Rebuild v4 component/relation JSONL files from final_veri_seti.json."""
    build_script = ROOT / "Data-Debate" / "gold" / "build_v4_dataset.py"
    command = [sys.executable, str(build_script)]
    if skip_external:
        command.append("--skip-external")
    print(f"[Dönüştürme] {' '.join(command)}", flush=True)
    subprocess.check_call(command, cwd=ROOT)


async def async_main(args: argparse.Namespace) -> None:
    """Run dataset generation."""
    if not OPENROUTER_API_KEY:
        raise SystemExit("HATA: .env dosyasında OPENROUTER_API_KEY tanımlı değil.")

    dataset = load_dataset(args.output)
    existing_texts = {
        normalized_text(item.get("data", {}).get("text", ""))
        for item in dataset
        if item.get("data", {}).get("text")
    }
    print(
        f"OpenRouter dataset üreticisi. Model={DEFAULT_MODEL}; mevcut={len(dataset)}; "
        f"hedef={args.target_count}",
        flush=True,
    )

    if len(dataset) >= args.target_count:
        print("Hedef örnek sayısı zaten tamamlanmış.", flush=True)
        if not args.skip_build:
            build_training_files(args.skip_external_build)
        return

    completed_topics = {
        item.get("data", {}).get("topic", "")
        for item in dataset
        if item.get("data", {}).get("topic")
    }
    topic_pool = [topic for topic in [*BASE_TOPICS, *EXPANSION_TOPICS] if topic not in completed_topics]
    if not topic_pool:
        raise SystemExit("Üretilecek yeni konu kalmadı; EXPANSION_TOPICS listesini genişlet.")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=200.0) as client:
        tasks = [
            process_topic(
                client,
                topic,
                sem,
                dataset,
                existing_texts,
                lock,
                args.target_count,
                args.examples_per_topic,
            )
            for topic in topic_pool
        ]
        await asyncio.gather(*tasks)

    print(
        f"\nTamamlandı. Toplam {len(dataset)} örnek {args.output} dosyasına kaydedildi.",
        flush=True,
    )
    print(f"Component dağılımı: {dict(component_counts(dataset))}", flush=True)
    print(f"Relation dağılımı: {dict(relation_counts(dataset))}", flush=True)

    if len(dataset) < args.target_count:
        raise SystemExit(
            f"Hedefe ulaşılamadı: {len(dataset)}/{args.target_count}. "
            "Daha fazla topic ekleyip scripti tekrar çalıştır."
        )
    if not args.skip_build:
        build_training_files(args.skip_external_build)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--examples-per-topic", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-external-build", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
