"""Validate generated rows against the approved rules and column specifications."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from synthgen.core.models import SchemaProfile
from synthgen.rules.expressions import compile_expression


@dataclass
class ValidationResult:
    """Pass/fail per row, plus failure counts per rule for the quality report."""

    passed: pd.Series
    failures_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def n_failed(self) -> int:
        return int((~self.passed).sum())


def validate(df: pd.DataFrame, profile: SchemaProfile) -> ValidationResult:
    """Check every row against column specs and every approved rule."""
    passed = pd.Series(True, index=df.index)
    failures: dict[str, int] = {}

    for col in profile.columns:
        if col.name not in df.columns:
            raise KeyError(f"Generated data is missing column '{col.name}'")
        ok = _column_checks(df[col.name], col)
        failures[f"column:{col.name}"] = int((~ok).sum())
        passed &= ok

    for rule in profile.rules:
        check = compile_expression(rule.expr)(df)
        if rule.when:
            applies = compile_expression(rule.when)(df)
            check = check | ~applies
        failures[rule.id] = int((~check).sum())
        passed &= check

    return ValidationResult(passed=passed, failures_by_rule=failures)


def _column_checks(series: pd.Series, col) -> pd.Series:
    ok = pd.Series(True, index=series.index)
    present = series.notna()
    if not col.nullable:
        ok &= present
    if col.pattern:
        ok &= ~present | series.astype(str).str.fullmatch(col.pattern).fillna(False)
    if col.allowed_values:
        ok &= ~present | series.isin(col.allowed_values)
    if col.type in {"integer", "decimal"}:
        numeric = pd.to_numeric(series, errors="coerce")
        if col.min is not None:
            ok &= ~present | (numeric >= float(col.min))
        if col.max is not None:
            ok &= ~present | (numeric <= float(col.max))
    if col.unique:
        ok &= ~(present & series.duplicated(keep=False))
    return ok.fillna(False).astype(bool)
