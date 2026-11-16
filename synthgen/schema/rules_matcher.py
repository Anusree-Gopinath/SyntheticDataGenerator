"""Method 1 of 3: the rules baseline for column understanding.

Matches normalised headers against concept names and synonyms (exact, then fuzzy) and
detects the data type from sample values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd

from synthgen.schema.dictionary import ConceptDictionary
from synthgen.schema.normalise import normalise_header

FUZZY_THRESHOLD = 0.85


@dataclass
class ColumnInference:
    column: str
    concept: str | None
    type: str | None
    confidence: float
    method: str


class RulesMatcher:
    method = "rules"

    def __init__(self, dictionary: ConceptDictionary):
        self.dictionary = dictionary
        self._names: list[tuple[str, str]] = []  # (normalised name, concept)
        for concept in dictionary:
            self._names.append((normalise_header(concept.concept), concept.concept))
            for syn in concept.synonyms:
                self._names.append((normalise_header(syn), concept.concept))

    def infer(self, header: str, values: pd.Series | None = None) -> ColumnInference:
        norm = normalise_header(header)
        best_concept, best_score = None, 0.0
        for name, concept in self._names:
            score = 1.0 if name == norm else SequenceMatcher(None, name, norm).ratio()
            if score > best_score:
                best_concept, best_score = concept, score
        if best_score < FUZZY_THRESHOLD:
            best_concept = None
        detected = detect_type(values) if values is not None and len(values) else None
        concept_type = self.dictionary.get(best_concept).type if best_concept else None
        return ColumnInference(
            column=header,
            concept=best_concept,
            type=detected or concept_type,
            confidence=round(best_score if best_concept else 0.0, 3),
            method=self.method,
        )


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_INT = re.compile(r"^-?\d+$")
_DEC = re.compile(r"^-?\d+\.\d+$")
_BOOL = {"true", "false", "yes", "no", "y", "n"}


def detect_type(values: pd.Series) -> str | None:
    """Detect a column type from sample values (all non-empty values must agree)."""
    vals = [str(v).strip() for v in values.dropna() if str(v).strip()]
    if not vals:
        return None
    if all(_DATE.match(v) for v in vals):
        return "date"
    if all(v.lower() in _BOOL for v in vals):
        return "boolean"
    if all(_INT.match(v) for v in vals):
        return "integer"
    if all(_INT.match(v) or _DEC.match(v) for v in vals):
        return "decimal"
    return "string"
