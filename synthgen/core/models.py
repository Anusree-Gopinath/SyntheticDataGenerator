"""Data model: schema profiles, columns, rules and run records.

These structures carry everything between components and are stored as YAML or JSON
so they are readable, versionable and easy to compare between runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

COLUMN_TYPES = {"string", "category", "integer", "decimal", "date", "boolean", "text"}
RULE_TYPES = {
    "format",
    "range",
    "allowed_values",
    "uniqueness",
    "temporal_order",
    "conditional_presence",
    "conditional_value",
}


@dataclass
class ColumnSpec:
    """One column in a schema profile."""

    name: str
    concept: str | None = None
    type: str = "string"
    pattern: str | None = None
    min: float | str | None = None
    max: float | str | None = None
    decimals: int | None = None
    allowed_values: list[str] | None = None
    unique: bool = False
    nullable: bool = False
    inferred_by: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in COLUMN_TYPES:
            raise ValueError(f"Unknown column type '{self.type}' for column '{self.name}'")


@dataclass
class RuleSpec:
    """A cross-column business rule written in the safe rule language."""

    id: str
    type: str
    description: str
    expr: str
    when: str | None = None
    source: str = "user"  # dictionary | llm | user
    approved: bool = False

    def __post_init__(self) -> None:
        if self.type not in RULE_TYPES:
            raise ValueError(f"Unknown rule type '{self.type}' for rule '{self.id}'")


@dataclass
class SchemaProfile:
    """The approved contract for generation. Only approved profiles drive generators."""

    profile_id: str
    version: int = 1
    columns: list[ColumnSpec] = field(default_factory=list)
    rules: list[RuleSpec] = field(default_factory=list)
    approved_by: str | None = None
    approved_at: str | None = None

    @property
    def is_approved(self) -> bool:
        return self.approved_by is not None and all(r.approved for r in self.rules)

    def column(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaProfile:
        return cls(
            profile_id=data["profile_id"],
            version=data.get("version", 1),
            columns=[ColumnSpec(**c) for c in data.get("columns", [])],
            rules=[RuleSpec(**r) for r in data.get("rules", [])],
            approved_by=data.get("approved_by"),
            approved_at=data.get("approved_at"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> SchemaProfile:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


@dataclass
class RunRecord:
    """One record per generator, scenario and seed. Logged for reproducibility."""

    run_id: str
    profile: str
    scenario: int
    approach: str
    seed: int
    rows_requested: int
    rows_rejected: int = 0
    llm_model: str | None = None
    code_version: str | None = None
    duration_seconds: float | None = None
    complete: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
