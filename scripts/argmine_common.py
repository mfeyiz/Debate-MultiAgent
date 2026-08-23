#!/usr/bin/env python3
"""Shared helpers for the v5 argument-mining data pipeline.

Used by generate_dataset_v5.py, consensus_label.py and build_gold_holdout.py.

Public label schema (fixed, matches production model):
- components: claim / evidence / other
- relations:  support / attack / none
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

COMPONENT_LABELS = {"claim", "evidence", "other"}
RELATION_LABELS = {"support", "attack", "none"}


def env_models(name: str, default: str) -> list[str]:
    """Read a comma-separated model list from the environment."""
    raw = os.getenv(name, default)
    return [m.strip() for m in raw.split(",") if m.strip()]


GENERATOR_MODELS = env_models("GENERATOR_MODELS", "deepseek/deepseek-v4-pro,openai/gpt-4o-mini")
JUDGE_MODELS = env_models(
    "JUDGE_MODELS", "meta-llama/llama-3.3-70b-instruct,qwen/qwen-2.5-72b-instruct"
)
HOLDOUT_GENERATOR_MODEL = os.getenv(
    "HOLDOUT_GENERATOR_MODEL", "mistralai/mistral-small-3.2-24b-instruct"
)

_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://logos-debate.app",
    "X-Title": "Logos Argument-Mining Pipeline",
}


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
def clean_response(text: str) -> str:
    """Strip optional markdown fences around a JSON response."""
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def normalized_text(text: str) -> str:
    """Duplicate-detection key for free text."""
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def normalized_topic(topic: str) -> str:
    """Duplicate-detection key for topic strings."""
    topic = normalized_text(topic)
    topic = re.sub(r"[^\wığüşöçİĞÜŞÖÇ]+", " ", topic, flags=re.UNICODE)
    return re.sub(r"\s+", " ", topic).strip()


def text_contains(haystack: str, needle: str) -> bool:
    """Case-insensitive containment with whitespace normalization."""
    return normalized_text(needle) in normalized_text(haystack)


def parse_json_loose(content: str) -> Any:
    """Parse a model JSON reply, tolerating a wrapping object or fences."""
    parsed = json.loads(clean_response(content))
    if isinstance(parsed, dict):
        for key in ("samples", "items", "data", "results", "paragraphs"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    return parsed


# --------------------------------------------------------------------------- #
# OpenRouter call
# --------------------------------------------------------------------------- #
async def call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_retries: int = 3,
    force_json: bool = False,
    timeout: float = 75.0,
    label: str = "",
) -> str | None:
    """Call OpenRouter chat completions and return raw message content."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = await client.post(
                OPENROUTER_API_BASE, headers=_HEADERS, json=payload, timeout=timeout
            )
            # Some OpenRouter providers (e.g. certain qwen routes) reject
            # response_format=json_object with a 400. Drop it and retry; the
            # callers parse JSON leniently anyway.
            if response.status_code == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                continue
            response.raise_for_status()
            body = response.json()
            msg = body["choices"][0]["message"]
            # Some reasoning models return content=null and put the answer in
            # the reasoning field; fall back to it so the JSON can still be parsed.
            content = msg.get("content") or msg.get("reasoning") or ""
            if content:
                return content
            continue  # empty completion: retry
        except Exception as exc:  # noqa: BLE001 - retry context for batch jobs
            wait = 2**attempt + 1
            print(
                f"      [retry {attempt + 1}/{max_retries}] {label or model}: {exc}",
                flush=True,
            )
            await asyncio.sleep(wait)
    return None


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
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
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Topic pool (debate topics for generation)
# --------------------------------------------------------------------------- #
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
    "Enflasyonla mücadelede faiz politikasının etkinliği",
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
    "Dört günlük çalışma haftasının üretkenlik ve ücretler üzerindeki etkisi",
    "Çocuklara akıllı telefon kullanım yaş sınırı getirilmesi",
    "Kamu kurumlarında yapay zeka destekli karar sistemlerinin kullanılması",
    "Online sınavlarda kamera gözetiminin mahremiyetle ilişkisi",
    "Kira artış sınırlamalarının konut arzına etkisi",
    "Turizm bölgelerinde kısa dönem kiralamanın yerel halk üzerindeki etkisi",
    "Kentsel dönüşüm projelerinde yerinde dönüşüm zorunluluğu",
    "Deprem sigortasının zorunlu kapsamının genişletilmesi",
    "Gıda israfını azaltmak için marketlere bağış zorunluluğu getirilmesi",
    "Meslek liselerinin teknoloji sektörüne ara eleman yetiştirme kapasitesi",
    "Dil öğreniminde yapay zeka sohbet araçlarının kullanılması",
    "Sahte haberle mücadelede devlet denetiminin ifade özgürlüğüne etkisi",
    "Seçim kampanyalarında deepfake kullanımına cezai yaptırım",
    "Uyuşturucu bağımlılığıyla mücadelede cezalandırma yerine tedavi yaklaşımı",
    "Nadir hastalık ilaçlarının kamu tarafından karşılanması",
    "Gig ekonomisi çalışanlarının sosyal güvence kapsamına alınması",
    "Merkez bankası dijital parasının bankacılık sistemine etkisi",
    "Laboratuvarda üretilen etin çevresel ve ekonomik etkileri",
    "Uydu internet hizmetlerinin kırsal dijital eşitsizliği azaltması",
    "Evrensel temel gelirin çalışma motivasyonu üzerindeki etkisi",
]

_DOMAINS = [
    ("Eğitim", ["devlet liseleri", "köy okulları", "üniversite hazırlık sınıfları", "anaokulları"]),
    ("Sağlık", ["aile sağlığı merkezleri", "şehir hastaneleri", "ruh sağlığı klinikleri", "evde bakım hizmetleri"]),
    ("Ekonomi", ["küçük esnaf", "ihracatçı KOBİ'ler", "tarım kooperatifleri", "serbest çalışanlar"]),
    ("Çevre", ["kıyı kentleri", "sanayi bölgeleri", "tarımsal havzalar", "turizm beldeleri"]),
    ("Teknoloji", ["kamu kurumları", "e-ticaret platformları", "okul yönetim sistemleri", "hastane bilgi sistemleri"]),
    ("Ulaşım", ["metrobüs hatları", "hızlı tren koridorları", "bisiklet ağları", "kırsal servis hatları"]),
]
_POLICIES = [
    "zorunlu şeffaflık raporu", "kademeli teşvik modeli", "bağımsız denetim şartı",
    "veri paylaşımı sınırı", "gelir temelli destek sistemi", "algoritmik karar açıklaması",
    "risk sigortası zorunluluğu", "bölgesel pilot uygulama",
]
_TENSIONS = [
    "maliyet ve erişim dengesi", "mahremiyet ve verimlilik çatışması",
    "kısa vadeli bütçe yükü ve uzun vadeli tasarruf", "eşitlik ve bireysel tercih özgürlüğü",
    "güvenlik ve kullanıcı deneyimi", "rekabet gücü ve sosyal koruma",
]


def build_topic_pool(limit: int) -> list[str]:
    """Build a pool of unique, specific debate topics."""
    topics = list(BASE_TOPICS)
    seen = {normalized_topic(t) for t in topics}
    for policy in _POLICIES:
        for tension in _TENSIONS:
            for domain, contexts in _DOMAINS:
                for context in contexts:
                    topic = (
                        f"{domain} alanında {context} için {policy} uygulanmasının "
                        f"{tension} üzerindeki etkisi"
                    )
                    key = normalized_topic(topic)
                    if key in seen:
                        continue
                    seen.add(key)
                    topics.append(topic)
                    if len(topics) >= limit:
                        return topics
    return topics
