"""Shared quality policy for Turkish component labels.

The project uses three public component labels:

- claim: arguable stance, risk, benefit, recommendation, evaluation, or result.
- evidence: data, example, research finding, observation, sourced reason, or statistic.
- other: non-argument context such as boilerplate, method notes, definitions, or history.

This module intentionally stays deterministic. It is not a replacement for model
training or human review; it is a conservative quality gate for obvious label drift
in generated and semi-synthetic datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


COMPONENT_LABELS = {"claim", "evidence", "other"}

_TRANSLATION = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ş": "s",
        "Ş": "s",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }
)


def normalize_text(text: str) -> str:
    """Return a Turkish-aware normalized string for rule matching."""
    normalized = str(text or "").translate(_TRANSLATION).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


META_RE = re.compile(
    r"\b("
    r"ana sayfa|cerez|gizlilik politikasi|site haritasi|basvuru formu|iletisim bilgileri|telefon|"
    r"e-posta|uyelik|abonelik|kategori|arsiv|paylasim dugmesi|kaynakca|"
    r"yontem notu|metodoloji|kapsam|orneklem|katilimci|takvim|gundem|"
    r"tablo aciklamasi|sekil aciklamasi|teknik ek|veri indirme|lisans|"
    r"tanim|terim|tarihce|adres|logo|sponsor|oturum|tema tercihi|"
    r"bu bolumde|giris bolumunde|ek bolumde|eklerinde|sayfanin sonunda|"
    r"ayri bolumde|listelen|aciklanmaktadir|yer almaktadir|bulunmaktadir"
    r")\b"
)

EVIDENCE_RE = re.compile(
    r"\b("
    r"ornegin|ornek olarak|verilerine gore|verilere gore|kaynaklara gore|rapora gore|"
    r"arastirma|calisma|anket|rapor|istatistik|deney|pilot|bulgu|"
    r"gozlem|kaynak|tuik|tcmb|oecd|sgk|unesco|dunya bankasi|bakanlik|"
    r"universite|hakemli|resmi|bulten|tablo|goster\w*|buldu|"
    r"belirt(?:ti|ildi|iyor|mistir|ilmistir|mektedir|mis)|"
    r"bildir(?:di|ildi|iyor|mistir|ilmistir|mektedir|mis)|"
    r"raporla\w*|olcul\w*|tespit edil\w*|"
    r"kanitla\w*|sonuclari|oran|yuzde|%"
    r")\b"
)

CLAIM_RE = re.compile(
    r"\b("
    r"artir\w*|arttir\w*|azalt\w*|dusur\w*|yukselt\w*|guclendir\w*|"
    r"zayiflat\w*|kolaylastir\w*|zorlastir\w*|destekle\w*|engelle\w*|"
    r"sagla\w*|yol ac\w*|neden ol\w*|tetikle\w*|"
    r"risk|tehdit|fayda|zarar|avantaj|dezavantaj|maliyet|"
    r"gerek\w*|gereklidir|gerektirir|olmali|olmamali|savunulabilir|"
    r"uygundur|uygun degildir|mumkun degildir|yerini alamaz|umut vadet|"
    r"onemlidir|hayati|etkilidir|verimlidir|maliyetlidir|yuksek\w*|"
    r"dusuk\w*|sinirli\w*|riskli\w*|basarili\w*|basarisiz\w*|"
    r"adil degildir|seffaf degildir|guvenilir degildir|mahremiyet|"
    r"kaliteyi|basariyi|motivasyonu|iletisimi|asiri|asıl sorun|asil sorun|"
    r"kaldirilmali|yasaklanmali|yasaklanmasi|zorunlu|talep etti|"
    r"anlamli olur|olumsuz etkiler|olumlu etkiler|derinlestir|"
    r"azalir|artar|dusmesi|yukselmesi|bekleniyor|muhtemeldir|"
    r"nasil soylenebilir"
    r")\b"
)

WEAK_EVIDENCE_WITHOUT_SOURCE_RE = re.compile(
    r"\b("
    r"olabilir|saglayabilir|artirabilir|azaltabilir|tetikleyebilir|"
    r"bekleniyor|muhtemeldir|risk tasir|zarar veriyor|fayda saglar|"
    r"avantaj saglayabilir|bir arac olabilir"
    r")\b"
)

STRONG_SOURCE_RE = re.compile(
    r"\b("
    r"verilerine gore|verilere gore|rapora gore|arastirmaya gore|calismaya gore|"
    r"ankete gore|resmi|tuik|tcmb|oecd|sgk|unesco|dunya bankasi|bakanlik|"
    r"universite|hakemli|bulten|tablo|pilot|deney|orneklem|katilimci|"
    r"olcul\w*|tespit edil\w*|raporla\w*|bildir\w*|belirt\w*|goster\w*|"
    r"buldu|sonuclari|oran|yuzde|%"
    r")\b"
)

OBSERVED_EVIDENCE_RE = re.compile(
    r"\b("
    r"verilerine gore|verilere gore|rapora gore|arastirmaya gore|calismaya gore|"
    r"ankete gore|resmi|tuik|tcmb|oecd|sgk|unesco|dunya bankasi|bakanlik|"
    r"universite|hakemli|bulten|tablo|olcul\w*|tespit edil\w*|"
    r"raporlad\w*|bildir\w*|belirt\w*|goster\w*|buldu|kanitla\w*|kanitlan\w*|"
    r"ortaya koy\w*|"
    r"sonuclari"
    r")\b"
)

BACKGROUND_RE = re.compile(
    r"\b("
    r"gelistirme asamasindadir|laboratuvar ortaminda test edilmektedir|"
    r"farkli yaklasimlar bulunmaktadir|yasal duzenleme bulunmamaktadir|"
    r"tarihsel olarak|olarak tanimlanir|ifade eder|hedefler|belirlenir"
    r")\b"
)


@dataclass(frozen=True)
class ComponentDecision:
    """Quality decision for one component row."""

    action: str
    label: str | None
    reasons: tuple[str, ...]


def normalize_label(label: str) -> str:
    """Normalize legacy component labels to the public label schema."""
    label_lower = str(label or "").strip().lower()
    return "other" if label_lower == "background" else label_lower


def infer_component_label(text: str) -> tuple[str | None, tuple[str, ...]]:
    """Infer the most likely component label from deterministic signals."""
    normalized = normalize_text(text)
    reasons: list[str] = []
    if len(normalized) < 8:
        return None, ("too_short",)

    has_meta = bool(META_RE.search(normalized))
    has_evidence = bool(EVIDENCE_RE.search(normalized) or re.search(r"\b\d+(?:[,.]\d+)?\b", normalized))
    has_claim = bool(CLAIM_RE.search(normalized))
    has_background = bool(BACKGROUND_RE.search(normalized))

    if has_meta:
        reasons.append("meta_context_signal")
    if has_evidence:
        reasons.append("evidence_signal")
    if has_claim:
        reasons.append("claim_signal")
    if has_background:
        reasons.append("background_signal")

    if has_meta and not has_claim:
        return "other", tuple(reasons)
    if has_evidence and not has_meta:
        if has_claim and WEAK_EVIDENCE_WITHOUT_SOURCE_RE.search(normalized) and not OBSERVED_EVIDENCE_RE.search(normalized):
            return "claim", tuple(reasons + ["weak_unsourced_evidence_is_claim"])
        if has_claim and not STRONG_SOURCE_RE.search(normalized) and not re.search(r"\b\d+(?:[,.]\d+)?\b", normalized):
            return "claim", tuple(reasons + ["claim_signal_overrides_weak_evidence"])
        return "evidence", tuple(reasons)
    if has_claim:
        return "claim", tuple(reasons)
    if has_background or has_meta:
        return "other", tuple(reasons)
    return None, tuple(reasons + ["no_clear_argument_signal"])


def audit_component_label(label: str, text: str) -> ComponentDecision:
    """Return keep/fix/drop decision for a component label."""
    normalized_label = normalize_label(label)
    if normalized_label not in COMPONENT_LABELS:
        return ComponentDecision("drop", None, (f"bad_label:{label}",))

    inferred, reasons = infer_component_label(text)
    if inferred is None:
        # Ambiguous generated text is useful as a hard example. Do not delete it
        # unless it has a structurally invalid label; surface the reason in the
        # audit report and keep the original supervision.
        return ComponentDecision("keep", normalized_label, reasons)
    if normalized_label == "claim" and inferred == "evidence" and "claim_signal" in reasons:
        # Attributed or quantified claims often contain words such as "rapor",
        # "belirtiyor", or a percentage while still expressing an arguable
        # conclusion. Do not weaken the user's label policy by converting those
        # stance-bearing claims into evidence automatically.
        return ComponentDecision("keep", normalized_label, reasons + ("claim_label_protected",))
    if inferred != normalized_label:
        return ComponentDecision("fix", inferred, reasons + (f"{normalized_label}_to_{inferred}",))
    return ComponentDecision("keep", normalized_label, reasons)


def semantic_label_error(label: str, text: str) -> str | None:
    """Compatibility helper for older scripts that expect an error string."""
    decision = audit_component_label(label, text)
    if decision.action == "keep":
        return None
    return ",".join(decision.reasons)
