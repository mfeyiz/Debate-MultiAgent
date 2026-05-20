"""Claim analysis utilities.

Provides manipulation detection, statistical claim flagging,
quote verification, and balanced reporting analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


class ClaimAnalyzer:
    """Analyze individual claims for manipulation, statistics, quotes, etc."""

    # --- Manipulation patterns (Turkish) ---
    CLICKBAIT_PATTERNS = [
        r"bomb[ae]\s*haber",
        r"şok\s*detay",
        r"gizli\s*kalması\s*gereken",
        r"kimse\s*konuşmuyor\s*ama",
        r"yıllardır\s*söylüyorduk",
        r"herkes\s*bunu\s*konuşuyor",
        r"aklınız\s*duracak",
        r"inanılmaz\s*gerçek",
        r"medya\s*bunu\s*yayınlamıyor",
    ]

    GENERALIZATION_PATTERNS = [
        r"hiç\s*kimse",
        r"herkes\s*biliyor",
        r"asla",
        r"kesinlikle",
        r"tamamen",
        r"hep",
        r"hiçbir\s*zaman",
        r"tümüyle",
    ]

    MISSING_SOURCE_PATTERNS = [
        r"kaynaklara\s*göre",
        r"iddialara\s*göre",
        r"belirtildiğine\s*göre",
        r"öğrenildi",
        r"edindiğimiz\s*bilgiye\s*göre",
        r"konuşan\s*kişiler",
        r"isminin\s*açıklanmasını\s*istemeyen",
    ]

    EMOTIONAL_CHARGED_TERMS = [
        "skandal", "şok", "ihanet", "rezalet", "felaket", "korkunç",
        "asla", "kesinlikle", "herkes", "hiç kimse", "yok ediyor",
        "dehşet", "facia", "kâbus", "utanç", "yıkım",
    ]

    # --- Quote patterns ---
    QUOTE_PATTERNS = [
        r'["""\']([^"""\']{10,200})["""\']',
        r'«([^»]{10,200})»',
    ]

    @classmethod
    def compute_claim_hash(cls, text: str) -> str:
        """Compute a stable hash for a claim text."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def compute_embedding_hash(cls, text: str) -> str:
        """Compute a simpler hash for approximate semantic matching."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        # Remove very common words
        stopwords = {"bir", "ve", "ile", "bu", "için", "çok", "daha", "gibi", "kadar", "olan"}
        words = [w for w in re.findall(r"[\wğüşöçıİĞÜŞÖÇ]+", normalized) if w not in stopwords and len(w) > 2]
        return hashlib.sha256(" ".join(sorted(words)).encode("utf-8")).hexdigest()[:32]

    @classmethod
    def analyze_manipulation(cls, text: str) -> dict[str, Any]:
        """Analyze text for manipulation signals."""
        lower = text.lower()

        clickbait_hits = sum(1 for p in cls.CLICKBAIT_PATTERNS if re.search(p, lower))
        generalization_hits = sum(1 for p in cls.GENERALIZATION_PATTERNS if re.search(p, lower))
        missing_source_hits = sum(1 for p in cls.MISSING_SOURCE_PATTERNS if re.search(p, lower))
        charged_hits = sum(1 for term in cls.EMOTIONAL_CHARGED_TERMS if term in lower)
        exclamation = min(text.count("!"), 3)
        all_caps = len(re.findall(r"\b[A-ZÇĞİÖŞÜ]{4,}\b", text))
        question_marks = min(text.count("?"), 3)

        # Combined score
        raw = (
            (clickbait_hits * 0.35)
            + (generalization_hits * 0.25)
            + (missing_source_hits * 0.20)
            + (charged_hits * 0.15)
            + (exclamation * 0.10)
            + (all_caps * 0.08)
            + (question_marks * 0.05)
        )
        score = min(1.0, raw)

        findings = []
        if clickbait_hits > 0:
            findings.append("Tıklama tuzağı (clickbait) dil sinyali")
        if generalization_hits > 0:
            findings.append("Aşırı genelleme")
        if missing_source_hits > 0:
            findings.append("Belirsiz kaynak atıfı")
        if charged_hits > 0:
            findings.append("Duygusal/manipülatif kelime kullanımı")
        if exclamation >= 2:
            findings.append("Aşırı ünlem kullanımı")

        return {
            "score": score,
            "clickbait_hits": clickbait_hits,
            "generalization_hits": generalization_hits,
            "missing_source_hits": missing_source_hits,
            "charged_hits": charged_hits,
            "findings": findings,
        }

    @classmethod
    def detect_statistical_claim(cls, text: str) -> dict[str, Any]:
        """Detect if a claim contains statistical data."""
        patterns = {
            "percentage": r"%?\s*\d+[,.]?\d*\s*%",
            "year_with_number": r"\b(19|20)\d{2}\b.*\d+[,.]?\d*",
            "number_with_unit": r"\d+[,.]?\d*\s*(?:milyon|milyar|bin|trilyon|TL|dolar|euro)",
            "rate_pattern": r"(?:oranı|yüzdesi|oran|oranını)\s*:?\s*(%?\s*\d+[,.]?\d*)",
        }

        found = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                found[key] = matches

        is_statistical = len(found) > 0
        extracted_numbers = re.findall(r"\d+[,.]?\d*", text)

        return {
            "is_statistical": is_statistical,
            "patterns_found": found,
            "extracted_numbers": extracted_numbers,
            "number_count": len(extracted_numbers),
        }

    @classmethod
    def detect_quotes(cls, text: str) -> list[dict[str, Any]]:
        """Detect quoted text and attempt to identify the speaker."""
        quotes = []
        for pattern in cls.QUOTE_PATTERNS:
            for match in re.finditer(pattern, text):
                quote_text = match.group(1).strip()
                # Try to find speaker attribution before the quote
                before = text[:match.start()]
                speaker = None
                speaker_patterns = [
                    r"([A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+(?:dedi|söyledi|belirtti|kaydetti)",
                    r"([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+(?:dedi|söyledi|belirtti)",
                    r"(?:['\"])\s*([A-ZÇĞİÖŞÜ][^'\"]{3,50})\s*:\s*['\"]",
                ]
                for sp in speaker_patterns:
                    sm = re.search(sp, before)
                    if sm:
                        speaker = sm.group(1)
                        break

                quotes.append({
                    "quote": quote_text,
                    "speaker": speaker,
                    "has_attribution": speaker is not None,
                })
        return quotes

    @classmethod
    def analyze_balanced_reporting(cls, article_text: str, claims: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze whether the article presents balanced reporting."""
        lower = article_text.lower()

        # Check for counter-argument signals
        counter_signals = [
            "diğer yandan", "buna karşın", "ancak", "fakat", "ama",
            "karşı görüş", "eleştirenler", "muhalefet", "karşı çıkan",
            "önemli olan", "dikkate alınması", "göz ardı edilmemeli",
        ]
        counter_hits = sum(1 for s in counter_signals if s in lower)

        # Check source diversity (rough heuristic)
        domains_mentioned = set()
        for claim in claims:
            text = claim.get("text", "").lower()
            # Look for institution names
            institutions = [
                "tüik", "tcmb", "sağlık bakanlığı", "milli eğitim",
                "adalet bakanlığı", "cumhurbaşkanlığı", "tbmm",
                "chp", "ak parti", "mhp", "iyi parti", "hdp",
            ]
            for inst in institutions:
                if inst in text:
                    domains_mentioned.add(inst)

        # Echo chamber: if all claims lean same direction
        sentiment_indicators = {
            "positive": ["başarı", "kazanım", "gelişme", "artış", "olumlu"],
            "negative": ["başarısız", "kaybı", "gerileme", "azalış", "olumsuz"],
        }
        pos_count = sum(1 for w in sentiment_indicators["positive"] if w in lower)
        neg_count = sum(1 for w in sentiment_indicators["negative"] if w in lower)
        imbalance = abs(pos_count - neg_count) > 3

        score = min(100, counter_hits * 15 + len(domains_mentioned) * 10)

        findings = []
        if counter_hits < 2:
            findings.append("Karşı görüş veya alternatif perspektif sınırlı")
        if len(domains_mentioned) < 2:
            findings.append("Kaynak çeşitliliği düşük")
        if imbalance:
            findings.append("Tek yönlü tonlama (dengesiz duygusal eğilim)")

        return {
            "balance_score": score,
            "counter_argument_signals": counter_hits,
            "source_diversity_count": len(domains_mentioned),
            "sources_mentioned": sorted(domains_mentioned),
            "is_imbalanced": imbalance,
            "findings": findings,
        }
