"""Combine the three methods with a weighted vote and flag columns for review."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from synthgen.schema.rules_matcher import ColumnInference


@dataclass
class CombinedInference:
    column: str
    concept: str | None
    confidence: float
    methods_agree: int
    methods_total: int
    needs_review: bool


def combine(
    inferences: list[ColumnInference],
    weights: dict[str, float],
    review_threshold: float = 0.7,
) -> CombinedInference:
    """Weighted vote across methods for one column.

    ``weights`` come from each method's accuracy on the validation schemas.
    """
    if not inferences:
        raise ValueError("No inferences to combine")
    scores: dict[str | None, float] = defaultdict(float)
    total = 0.0
    for inf in inferences:
        w = weights.get(inf.method, 1.0)
        scores[inf.concept] += w * inf.confidence
        total += w
    concept = max(scores, key=scores.get)
    confidence = scores[concept] / total if total else 0.0
    agree = sum(1 for inf in inferences if inf.concept == concept)
    needs_review = (
        concept is None or confidence < review_threshold or agree < len(inferences)
    )
    return CombinedInference(
        column=inferences[0].column,
        concept=concept,
        confidence=round(confidence, 3),
        methods_agree=agree,
        methods_total=len(inferences),
        needs_review=needs_review,
    )
