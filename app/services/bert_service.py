"""ModernBERT pipeline service (real inference).

Loads two fine-tuned ModernBERT models:
1. Component extractor  – token-level BIO (claim / evidence).
2. Relation classifier  – sequence-level (support / attack / neutral).

Both models expect the Turkish ``modernbert-tr-base-1k`` tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer

from app.config import Config


@dataclass
class ArgumentComponent:
    """An extracted claim or evidence snippet."""

    text: str
    component_type: str  # "claim" | "evidence"
    start_idx: int
    end_idx: int


@dataclass
class Relation:
    """A relation between a source component and a target component."""

    source_component: ArgumentComponent
    target_component: ArgumentComponent
    relation_type: str  # "support" | "attack" | "neutral"
    confidence: float


@dataclass
class PipelineResult:
    """Output of the ModernBERT pipeline."""

    components: List[ArgumentComponent]
    relations: List[Relation]
    overall_strength: float  # 0.0 - 1.0
    feedback: str | None


class ModernBERTPipeline:
    """Real pipeline backed by two fine-tuned ModernBERT models."""

    STRENGTH_THRESHOLD: float = Config.EVIDENCE_STRENGTH_THRESHOLD

    _COMPONENT_LABELS = ["O", "B-CLAIM", "I-CLAIM", "B-EVIDENCE", "I-EVIDENCE"]
    _RELATION_LABELS = ["support", "attack", "neutral"]

    def __init__(
        self,
        component_model_dir: str | Path = "models/component_extractor/final",
        relation_model_dir: str | Path = "models/relation_classifier/final",
    ) -> None:
        self.device = self._get_device()

        comp_path = Path(component_model_dir)
        rel_path = Path(relation_model_dir)

        if not comp_path.exists():
            raise FileNotFoundError(
                f"Component extractor not found at {comp_path}. "
                "Run `python train_component_extractor.py` first."
            )
        if not rel_path.exists():
            raise FileNotFoundError(
                f"Relation classifier not found at {rel_path}. "
                "Run `python train_relation_classifier.py` first."
            )

        self.comp_tokenizer = AutoTokenizer.from_pretrained(comp_path)
        self.comp_model = AutoModelForTokenClassification.from_pretrained(comp_path)
        self.comp_model.to(self.device)
        self.comp_model.eval()

        self.rel_tokenizer = AutoTokenizer.from_pretrained(rel_path)
        self.rel_model = AutoModelForSequenceClassification.from_pretrained(rel_path)
        self.rel_model.to(self.device)
        self.rel_model.eval()

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def analyze(
        self,
        source_text: str,
        target_text: str,
        source_type: str = "claim",
        target_type: str = "evidence",
    ) -> PipelineResult:
        """Analyze *source_text* w.r.t *target_text*.

        In a debate exchange:
            target_text = the preceding claim
            source_text = the response (evidence / rebuttal / etc.)
        """
        # 1. Extract components from both texts
        all_components: List[ArgumentComponent] = []
        all_components.extend(self._extract_components(target_text, "claim"))
        all_components.extend(self._extract_components(source_text, "evidence"))

        # 2. Pair every claim with every evidence and classify relation
        relations: List[Relation] = []
        claims = [c for c in all_components if c.component_type == "claim"]
        evidences = [c for c in all_components if c.component_type == "evidence"]

        for claim in claims:
            for ev in evidences:
                rel_type, conf = self._classify_relation(claim.text, ev.text)
                relations.append(
                    Relation(
                        source_component=ev,
                        target_component=claim,
                        relation_type=rel_type,
                        confidence=conf,
                    )
                )

        # 3. Evaluate overall strength
        strength = self._evaluate_strength(source_text, target_text, relations)

        # 4. Generate diagnostic feedback (always)
        feedback = self._generate_feedback(source_text, target_text, relations, strength)

        return PipelineResult(
            components=all_components,
            relations=relations,
            overall_strength=strength,
            feedback=feedback,
        )

    # ------------------------------------------------------------------
    # Component extraction (token classification)
    # ------------------------------------------------------------------

    def _extract_components(self, text: str, default_type: str) -> List[ArgumentComponent]:
        """Run the component extractor and convert BIO tags to spans."""
        enc = self.comp_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = self.comp_model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=2)[0].cpu().tolist()

        spans: List[ArgumentComponent] = []
        current_start = None
        current_label = None

        for idx, (pred, (char_start, char_end)) in enumerate(zip(predictions, offsets)):
            label = self._COMPONENT_LABELS[pred]
            if label.startswith("B-"):
                if current_start is not None:
                    spans.append(
                        ArgumentComponent(
                            text=text[current_start:prev_end],
                            component_type=current_label.replace("B-", "").lower(),
                            start_idx=current_start,
                            end_idx=prev_end,
                        )
                    )
                current_start = char_start
                current_label = label
                prev_end = char_end
            elif label.startswith("I-") and current_start is not None:
                prev_end = char_end
            else:
                if current_start is not None:
                    spans.append(
                        ArgumentComponent(
                            text=text[current_start:prev_end],
                            component_type=current_label.replace("B-", "").lower(),
                            start_idx=current_start,
                            end_idx=prev_end,
                        )
                    )
                current_start = None
                current_label = None

        if current_start is not None:
            spans.append(
                ArgumentComponent(
                    text=text[current_start:prev_end],
                    component_type=current_label.replace("B-", "").lower(),
                    start_idx=current_start,
                    end_idx=prev_end,
                )
            )

        # If model produced no spans, fall back to a single full-text span
        if not spans:
            spans.append(
                ArgumentComponent(
                    text=text[:240],
                    component_type=default_type,
                    start_idx=0,
                    end_idx=min(len(text), 240),
                )
            )
        return spans

    # ------------------------------------------------------------------
    # Relation classification
    # ------------------------------------------------------------------

    def _classify_relation(self, claim_text: str, evidence_text: str) -> tuple[str, float]:
        text = f"{claim_text} [SEP] {evidence_text}"
        enc = self.rel_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.rel_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs))
        confidence = round(float(probs[pred_id]), 3)
        return self._RELATION_LABELS[pred_id], confidence

    # ------------------------------------------------------------------
    # Strength evaluation
    # ------------------------------------------------------------------

    def _evaluate_strength(
        self, source: str, target: str, relations: List[Relation]
    ) -> float:
        source_lower = source.lower()

        # Base: squish raw confidence to a narrower range so that
        # typical classifier confidence (0.80-0.95) maps to (0.40-0.60).
        # This makes room for genuine evidence richness to push scores up.
        if relations:
            raw_avg = sum(r.confidence for r in relations) / len(relations)
            base = (raw_avg - 0.5) * 0.6 + 0.5
        else:
            base = 0.30

        # Evidence richness bonus (Turkish keywords) — reduced per-hit value
        evidence_markers = [
            "rapor", "araştırma", "veri", "çalışma", "anket", "analiz",
            "istatistik", "örnek", " kanıt", "bulgu", "göstermektedir",
            "ortaya koymaktadır", "belirtmektedir", "vurgulamaktadır",
        ]
        hits = sum(1 for w in evidence_markers if w in source_lower)
        bonus = min(hits * 0.02, 0.06)

        # Length penalty — increased thresholds
        word_count = len(source.split())
        if word_count < 20:
            penalty = 0.20
        elif word_count < 40:
            penalty = 0.10
        else:
            penalty = 0.0

        # Penalise weak attacks (few evidence keywords)
        attack_rels = [r for r in relations if r.relation_type == "attack"]
        if attack_rels and hits < 2:
            penalty += 0.10

        # Penalise sparse relations (only 1-2 links detected)
        if len(relations) <= 1:
            penalty += 0.08

        # Penalise when all relations are neutral (no clear stance)
        if relations and all(r.relation_type == "neutral" for r in relations):
            penalty += 0.12

        # Penalise lack of numerical evidence
        has_numeric = any(m in source_lower for m in ["%", "yüzde", "oran", "sayı"])
        if not has_numeric:
            penalty += 0.05

        strength = base + bonus - penalty
        return round(max(0.0, min(1.0, strength)), 2)

    # ------------------------------------------------------------------
    # Turkish feedback
    # ------------------------------------------------------------------

    def _generate_feedback(
        self, source: str, target: str, relations: List[Relation], strength: float
    ) -> str:
        has_support = any(r.relation_type == "support" for r in relations)
        has_attack = any(r.relation_type == "attack" for r in relations)
        has_neutral = any(r.relation_type == "neutral" for r in relations)
        weak_threshold = self.STRENGTH_THRESHOLD - 0.10
        mid_threshold = self.STRENGTH_THRESHOLD - 0.05

        feedbacks = []

        if strength >= self.STRENGTH_THRESHOLD:
            return "Argüman güçlü ve yeterli şekilde desteklenmiş."

        if has_neutral and not (has_support or has_attack):
            feedbacks.append(
                "Bu yanıt iddiayla net bir destek veya çürütme ilişkisi kurmuyor. "
                "Lütfen tarafınızı netleştirin ve hedefe yönelik somut gerekçeler sunun."
            )

        if has_attack and strength < mid_threshold:
            feedbacks.append(
                "Saldırı doğrudan iddianın özünü hedef almıyor. "
                "Karşıt görüşü çürüten ampirik veri veya mantıksal çelişki örneği ekleyin."
            )

        if has_support and strength < mid_threshold:
            feedbacks.append(
                "Destekleyici kanıt zayıf veya dolaylı kalıyor. "
                "Veri ile iddia arasındaki nedensel bağlantıyı daha açık kurun."
            )

        if has_attack and weak_threshold <= strength < self.STRENGTH_THRESHOLD:
            feedbacks.append(
                "Saldırı yaklaşık olarak hedefe yönelik ancak daha kesin bir çelişki noktası sunulabilir."
            )

        if has_support and weak_threshold <= strength < self.STRENGTH_THRESHOLD:
            feedbacks.append(
                "Destek yaklaşık olarak yeterli ancak nedensel bağ daha net kurulabilir."
            )

        if not any(m in source.lower() for m in ["%", "yüzde", "oran", "sayı"]):
            feedbacks.append(
                "Argüman sayısal veri içermiyor. İstatistik, anket sonucu veya karşılaştırmalı "
                "veri ekleyerek kanıtın gücünü artırabilirsiniz."
            )

        if len(source.split()) < 25:
            feedbacks.append(
                "Yanıt çok kısa. Daha detaylı açıklama ve birden fazla kaynakla "
                "desteklenmiş argüman sunmanız önerilir."
            )

        if not feedbacks:
            feedbacks.append(
                "Argüman genel hatlarıyla kabul edilebilir ancak daha kesin ve "
                "kapsamlı kanıtlarla güçlendirilebilir."
            )

        return " ".join(feedbacks)
