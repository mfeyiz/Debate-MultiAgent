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
            "percentage": r"(?:%\s*\d+[,.]?\d*|\d+[,.]?\d*\s*%|yüzde\s+\d+[,.]?\d*)",
            "year_with_number": r"\b(19|20)\d{2}\b.*\d+[,.]?\d*",
            "number_with_unit": r"\d+[,.]?\d*\s*(?:milyon|milyar|bin|trilyon|TL|dolar|euro)",
            "rate_pattern": r"(?:oranı|yüzdesi|oran|oranını)\s*:?\s*(?:%?\s*\d+[,.]?\d*|yüzde\s+\d+[,.]?\d*)",
            "economic_rate": r"(?:enflasyon|büyüme|faiz|işsizlik|kur|dolar|euro).*(?:%|yüzde|\d+[,.]?\d*)",
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

    @classmethod
    async def analyze_argument_structure(
        cls,
        article_text: str,
        claims: list[Any],
        model: Any | None = None,
    ) -> dict[str, Any]:
        """Analyze article text for thesis, logical fallacies, and dialectic quality."""
        if model is not None:
            try:
                from pydantic_ai import Agent as PydanticAgent
                
                # Setup structured prompt
                prompt = (
                    "Aşağıdaki haber metnini argüman madenciliği ve mantıksal tutarlılık açısından incele.\n"
                    "Metnin ana tezini (main thesis), genel retorik tonunu (dialectic tone), 0-100 arası mantıksal tutarlılık/objektiflik skorunu "
                    "ve metinde tespit ettiğin mantıksal safsataları (Logical Fallacies - örn. Ad Hominem, Saman Adam, "
                    "Otoriteye Başvuru, Korkuya Başvuru, Kısır Döngü vb.) Türkçe olarak çıkar.\n\n"
                    f"Haber Metni:\n{article_text}\n"
                )
                
                agent = PydanticAgent(
                    model,
                    result_type=ArgumentStructureResult,
                    system_prompt=(
                        "Sen profesyonel bir argüman madenciliği araştırmacısı ve medya okuryazarlığı editörüsün. "
                        "Haber metinlerindeki mantıksal yapıları ve safsataları titizlikle analiz edersin."
                    ),
                    model_settings={"temperature": 0.15},
                )
                
                response = await agent.run(prompt)
                res_obj = response.output
                return {
                    "main_thesis": res_obj.main_thesis,
                    "dialectic_tone": res_obj.dialectic_tone,
                    "objectivity_score": res_obj.objectivity_score,
                    "fallacies": [
                        {
                            "title": item.title,
                            "detail": item.detail,
                            "severity": item.severity,
                            "confidence": item.confidence,
                        }
                        for item in res_obj.fallacies
                    ]
                }
            except Exception as exc:
                import sys
                print(f"LLM argument mining analysis failed, falling back to rule-based: {exc}", file=sys.stderr)

        # Fallback rule-based analysis
        lower = article_text.lower()
        clickbait_hits = sum(1 for p in cls.CLICKBAIT_PATTERNS if re.search(p, lower))
        generalization_hits = sum(1 for p in cls.GENERALIZATION_PATTERNS if re.search(p, lower))
        missing_source_hits = sum(1 for p in cls.MISSING_SOURCE_PATTERNS if re.search(p, lower))
        charged_hits = sum(1 for term in cls.EMOTIONAL_CHARGED_TERMS if term in lower)

        fallacies = []
        if clickbait_hits > 0:
            fallacies.append({
                "title": "Tıklama Tuzağı ve Sansasyonellik",
                "detail": "Metin, okuyucunun merakını veya duygularını manipüle etmeye yönelik abartılı başlık/ifadeler içeriyor.",
                "severity": "medium",
                "confidence": 0.85
            })
        if generalization_hits > 0:
            fallacies.append({
                "title": "Aşırı Genelleme (Hasty Generalization)",
                "detail": "Metin, 'herkes', 'asla', 'hiçbir zaman' gibi sözcüklerle sınırlı gözlemlerden genel yargılara varıyor.",
                "severity": "medium",
                "confidence": 0.80
            })
        if missing_source_hits > 0:
            fallacies.append({
                "title": "Kanıtsız İddia ve Belirsiz Kaynak",
                "detail": "İddialar somut kanıtlara dayandırılmak yerine 'iddia edildi', 'kaynaklara göre' denilerek belirsiz bırakılmış.",
                "severity": "high",
                "confidence": 0.90
            })
        if charged_hits >= 3:
            fallacies.append({
                "title": "Duygulara Başvurmak (Appeal to Emotion)",
                "detail": "Akılcı argümanlar sunmak yerine okuyucuda öfke, korku veya şok yaratacak kelimeler yoğun kullanılmış.",
                "severity": "high",
                "confidence": 0.75
            })

        # Infer basic tone
        if clickbait_hits > 0 or charged_hits >= 2:
            tone = "Sansasyonel ve Alarmist"
            score = max(30, 90 - (clickbait_hits * 15 + charged_hits * 10))
        else:
            tone = "Göreceli Dengeli / Bilgilendirici"
            score = min(100, 95 - (generalization_hits * 8 + missing_source_hits * 10))

        # Infer basic thesis from first paragraph
        first_paragraph = article_text.strip().split("\n")[0]
        thesis = first_paragraph[:160] + "..." if len(first_paragraph) > 160 else first_paragraph

        return {
            "main_thesis": thesis or "Haber tezi analiz edilemedi.",
            "dialectic_tone": tone,
            "objectivity_score": int(score),
            "fallacies": fallacies
        }


# --- Pydantic models for structured output ---
from pydantic import BaseModel, Field

class FallacyItem(BaseModel):
    title: str = Field(description="Name of the fallacy in Turkish (e.g. Ad Hominem, Saman Adam, Otoriteye Başvuru)")
    detail: str = Field(description="Detailed explanation in Turkish of where it occurs in the text and why it is a fallacy.")
    severity: str = Field(description="Severity: high, medium, or low")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class ArgumentStructureResult(BaseModel):
    main_thesis: str = Field(description="The core main thesis or claim of the news article in Turkish.")
    dialectic_tone: str = Field(description="The rhetorical/dialectic tone description in Turkish, e.g. Objektif, Sansasyonel, Yanlı, Korku Odaklı.")
    objectivity_score: int = Field(description="Overall objectivity and logical consistency score out of 100.")
    fallacies: list[FallacyItem] = Field(description="List of logical fallacies identified in the text.")
