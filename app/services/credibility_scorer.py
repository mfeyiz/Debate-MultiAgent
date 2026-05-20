"""Turkish media source credibility and bias scoring.

Provides a local database of Turkish media outlets with
reliability scores and political bias labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SourceProfile:
    """Credibility profile for a media domain."""

    domain: str
    credibility_score: float  # 0-100
    bias: str  # left / center-left / center / center-right / right / unknown
    fact_checking_score: float  # 0-100, how often they correct errors
    transparency_score: float  # 0-100, how transparent about sources/methods
    notes: str


# Local Turkish media credibility database
# Scores are illustrative; in production these should be crowd-sourced or
# derived from an external dataset.
TURKISH_MEDIA_DB: dict[str, SourceProfile] = {
    "trthaber.com": SourceProfile(
        "trthaber.com", 78, "center", 75, 80,
        "TRT Haber - kamu yayıncısı, resmi verilere yakın"
    ),
    "aa.com.tr": SourceProfile(
        "aa.com.tr", 76, "center", 72, 82,
        "Anadolu Ajansı - kamu ajansı, hızlı haber kaynağı"
    ),
    "hurriyet.com.tr": SourceProfile(
        "hurriyet.com.tr", 68, "center-right", 55, 60,
        "Hürriyet - eski yaygın gazete, dijitalde çeşitli yazarlar"
    ),
    "milliyet.com.tr": SourceProfile(
        "milliyet.com.tr", 66, "center-right", 52, 58,
        "Milliyet - Demirören Grubu"
    ),
    "sozcu.com.tr": SourceProfile(
        "sozcu.com.tr", 58, "left", 48, 50,
        "Sözcü - muhalif ton, kimi zaman duygusal dil"
    ),
    "t24.com.tr": SourceProfile(
        "t24.com.tr", 72, "center-left", 65, 70,
        "T24 - bağımsız haber sitesi, kaynak gösterme eğilimi yüksek"
    ),
    "diken.com.tr": SourceProfile(
        "diken.com.tr", 74, "center-left", 68, 72,
        "Diken - analitik habercilik, kaynakça kullanımı iyi"
    ),
    "cumhuriyet.com.tr": SourceProfile(
        "cumhuriyet.com.tr", 70, "left", 60, 65,
        "Cumhuriyet - sol-muhalif gelenek, kimi zaman ideolojik ton"
    ),
    "haberturk.com": SourceProfile(
        "haberturk.com", 68, "center", 58, 62,
        "Habertürk - Ciner Grubu, orta ton"
    ),
    "ntv.com.tr": SourceProfile(
        "ntv.com.tr", 74, "center", 65, 70,
        "NTV - Doğuş Grubu, profesyonel habercilik"
    ),
    "cnnturk.com": SourceProfile(
        "cnnturk.com", 70, "center", 60, 65,
        "CNN Türk - Demirören Grubu"
    ),
    "haber7.com": SourceProfile(
        "haber7.com", 52, "right", 40, 45,
        "Haber7 - muhafazakâr ton, kimi zaman duygusal başlıklar"
    ),
    "yenisafak.com": SourceProfile(
        "yenisafak.com", 48, "right", 38, 42,
        "Yeni Şafak - güçlü ideolojik ton, kaynakça zayıf"
    ),
    "odatv.com": SourceProfile(
        "odatv.com", 50, "left", 42, 48,
        "OdaTV - muhalif ton, iddialı haberler, kimi zaman kanıtsız"
    ),
    "tr.sputniknews.com": SourceProfile(
        "tr.sputniknews.com", 35, "unknown", 30, 35,
        "Sputnik Türkiye - Rus devlet ajansı, propaganda riski yüksek"
    ),
    "tr.euronews.com": SourceProfile(
        "tr.euronews.com", 80, "center", 78, 82,
        "Euronews Türkçe - AB finansmanlı, çok dilli kaynak"
    ),
    "bbc.com": SourceProfile(
        "bbc.com", 85, "center", 82, 85,
        "BBC - uluslararası kamu yayıncısı, yüksek editoryal standart"
    ),
    "reuters.com": SourceProfile(
        "reuters.com", 86, "center", 84, 86,
        "Reuters - haber ajansı, objektiflik kuralı"
    ),
    "apnews.com": SourceProfile(
        "apnews.com", 85, "center", 83, 85,
        "AP - haber ajansı, yüksek standart"
    ),
    # Fact-check archives
    "teyit.org": SourceProfile(
        "teyit.org", 90, "center", 92, 90,
        "Teyit - bağımsız doğrulama platformu, IFCN üyesi"
    ),
    "dogrulukpayi.com": SourceProfile(
        "dogrulukpayi.com", 88, "center", 90, 88,
        "Doğruluk Payı - IFCN üyesi Türk doğrulama platformu"
    ),
    # Knowledge bases
    "tuik.gov.tr": SourceProfile(
        "tuik.gov.tr", 95, "center", 95, 95,
        "TÜİK - resmi istatistik kurumu"
    ),
    "tcmb.gov.tr": SourceProfile(
        "tcmb.gov.tr", 95, "center", 95, 95,
        "TCMB - Merkez Bankası resmi verileri"
    ),
    "resmigazete.gov.tr": SourceProfile(
        "resmigazete.gov.tr", 98, "center", 98, 98,
        "Resmi Gazete - yasal metinlerin tek yetkili kaynağı"
    ),
    "who.int": SourceProfile(
        "who.int", 90, "center", 90, 90,
        "Dünya Sağlık Örgütü"
    ),
    "worldbank.org": SourceProfile(
        "worldbank.org", 88, "center", 88, 88,
        "Dünya Bankası"
    ),
}


class CredibilityScorer:
    """Score the credibility of a source domain."""

    @classmethod
    def get_profile(cls, domain: str) -> SourceProfile | None:
        """Return the profile for a domain, or None if unknown."""
        normalized = domain.lower().replace("www.", "").replace("m.", "")
        # Exact match
        if normalized in TURKISH_MEDIA_DB:
            return TURKISH_MEDIA_DB[normalized]
        # Substring match
        for key, profile in TURKISH_MEDIA_DB.items():
            if key in normalized or normalized in key:
                return profile
        return None

    @classmethod
    def score_domain(cls, domain: str) -> dict[str, Any]:
        """Return a full credibility breakdown for a domain."""
        profile = cls.get_profile(domain)
        if profile:
            return {
                "known": True,
                "credibility_score": profile.credibility_score,
                "bias": profile.bias,
                "fact_checking_score": profile.fact_checking_score,
                "transparency_score": profile.transparency_score,
                "notes": profile.notes,
            }
        # Unknown domain - derive a rough heuristic
        return {
            "known": False,
            "credibility_score": 50,
            "bias": "unknown",
            "fact_checking_score": 40,
            "transparency_score": 40,
            "notes": "Bu kaynak yerel veritabanında tanımlı değil; dikkatli değerlendirin.",
        }

    @classmethod
    def get_bias_label(cls, domain: str) -> str:
        """Return the political bias label for a domain."""
        profile = cls.get_profile(domain)
        return profile.bias if profile else "unknown"
