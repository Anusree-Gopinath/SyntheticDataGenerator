"""The concept dictionary: the court concepts the system knows about."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Concept:
    concept: str
    description: str
    type: str
    synonyms: list[str] = field(default_factory=list)
    default_format: str | None = None
    pattern: str | None = None
    allowed_values: list[str] | None = None
    typical_rules: list[dict] = field(default_factory=list)


class ConceptDictionary:
    def __init__(self, concepts: list[Concept]):
        self.concepts = {c.concept: c for c in concepts}

    @classmethod
    def load(cls, path: str | Path) -> ConceptDictionary:
        data = yaml.safe_load(Path(path).read_text())
        return cls([Concept(**item) for item in data["concepts"]])

    def __iter__(self):
        return iter(self.concepts.values())

    def __len__(self) -> int:
        return len(self.concepts)

    def get(self, name: str) -> Concept:
        return self.concepts[name]
