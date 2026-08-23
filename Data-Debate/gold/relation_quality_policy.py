"""Deterministic quality policy for Turkish relation labels.

Relation semantics are harder to infer safely than component labels, so this
module focuses on conservative corrections:

- normalize the public labels,
- enforce graph schema constraints,
- detect duplicate pair conflicts,
- keep support/attack semantics unchanged unless the component labels make them
  structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

import component_quality_policy as cqp


RELATION_LABELS = {"support", "attack", "none"}


@dataclass(frozen=True)
class RelationDecision:
    """Quality decision for a relation row."""

    action: str
    label: str | None
    reasons: tuple[str, ...]


def normalize_label(label: str) -> str:
    """Normalize legacy relation labels to the public label schema."""
    label_lower = str(label or "").strip().lower()
    return "none" if label_lower == "neutral" else label_lower


def relation_key(claim_text: str, evidence_text: str) -> tuple[str, str]:
    """Normalized key for detecting conflicting duplicate relation pairs."""
    return (cqp.normalize_text(claim_text), cqp.normalize_text(evidence_text))


def audit_relation_label(
    label: str,
    *,
    source_component_label: str | None = None,
    target_component_label: str | None = None,
) -> RelationDecision:
    """Return keep/fix/drop decision for relation structure."""
    normalized_label = normalize_label(label)
    if normalized_label not in RELATION_LABELS:
        return RelationDecision("drop", None, (f"bad_label:{label}",))

    source_label = cqp.normalize_label(source_component_label or "")
    target_label = cqp.normalize_label(target_component_label or "")

    if source_label == "other" and normalized_label != "none":
        return RelationDecision("fix", "none", (f"{normalized_label}_from_other_to_none",))
    if normalized_label in {"support", "attack"} and target_label and target_label != "claim":
        return RelationDecision("drop", None, ("argumentative_relation_target_not_claim",))
    if normalized_label == "none" and source_label and source_label != "other":
        # This is project policy, not a universal argument-mining law. The v4
        # generated holdout/training schema defines none as the explicit other
        # component relation, so keep the training signal aligned with it.
        return RelationDecision("drop", None, ("none_relation_source_not_other",))
    return RelationDecision("keep", normalized_label, ())
