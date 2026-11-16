"""Shared column preparation for the learned approaches (Gaussian copula, CTGAN, LLM).

Each profile column a generator can learn becomes a ``Field``: the kind of values it holds
and the statistics it is allowed to use.

- Scenario 1: statistics come from the calibration store.
- Scenario 2: statistics come from the fictional sample. Columns the sample leaves empty
  fall back to the calibration store.

Dependent dates (for example a hearing after the issue date) are modelled as a gap in days
from their parent date, so the order between them is learned rather than left to chance.

Columns with nothing to learn from (identifiers, unique columns, and columns with no
statistics) are left to the knowledge-based generator, so every approach treats them alike
and the comparison is about what each approach learns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from synthgen.core.models import ColumnSpec, SchemaProfile
from synthgen.generators.base import CalibrationStats
from synthgen.generators.knowledge_based import KnowledgeBasedGenerator

EPOCH = pd.Timestamp("1970-01-01")
# Fewer present values than this in the sample and the column is not learned from the sample
MIN_OBSERVED = 5
_TEMPORAL = re.compile(r"^\s*(\w+)\s*>=?\s*(\w+)\s*$")


@dataclass
class Field:
    """One learnable column.

    ``kind`` is ``category``, ``number``, ``date`` (days since 1970) or ``gap`` (days after
    ``parent``). In Scenario 2, ``data`` holds the sample's values in those units, one per
    sample row, with ``None``/NaN where the row is empty or the value isn't allowed.
    ``dist`` holds the Scenario 1 description when there is no sample data.
    """

    name: str
    kind: str
    missing: float = 0.0  # share of rows to leave empty (0 for non-nullable columns)
    categories: list[str] = field(default_factory=list)
    probs: np.ndarray | None = None
    data: np.ndarray | None = None
    observed_missing: np.ndarray | None = None  # Scenario 2: True where the sample row is empty
    dist: dict | None = None
    parent: str | None = None
    integer: bool = False
    decimals: int = 2
    lo: float | None = None
    hi: float | None = None

    @property
    def learned_from_sample(self) -> bool:
        return self.data is not None

    def present_values(self) -> np.ndarray:
        """Scenario 2 values that are present (numbers, day counts or category names)."""
        if self.kind == "category":
            return np.array([v for v in self.data if v is not None], dtype=object)
        values = np.asarray(self.data, dtype=float)
        return values[~np.isnan(values)]

    def imputed(self, rng: np.random.Generator) -> np.ndarray:
        """Scenario 2 data with every gap filled by a random present value.

        Learners need a value in every row; whether the row was empty is modelled
        separately through ``observed_missing``.
        """
        present = self.present_values()
        if self.kind == "category":
            out = np.array(self.data, dtype=object)
            holes = np.array([v is None for v in out])
            out[holes] = rng.choice(self.categories, size=int(holes.sum()), p=self.probs)
            return out
        out = np.asarray(self.data, dtype=float).copy()
        holes = np.isnan(out)
        out[holes] = rng.choice(present, size=int(holes.sum()))
        return out


def is_identifier(col: ColumnSpec) -> bool:
    return col.unique or col.concept == "case_reference"


def temporal_pairs(profile: SchemaProfile) -> dict[str, str]:
    """child -> parent for approved rules of the form ``child > parent`` between dates."""
    dates = {c.name for c in profile.columns if c.type == "date"}
    pairs: dict[str, str] = {}
    for rule in profile.rules:
        match = _TEMPORAL.match(rule.expr)
        if rule.type == "temporal_order" and match:
            child, parent = match.groups()
            if child in dates and parent in dates and child != parent:
                pairs.setdefault(child, parent)
    return pairs


def to_days(values: pd.Series) -> np.ndarray:
    """Dates as float days since 1970 (NaN where empty or unreadable)."""
    parsed = pd.to_datetime(values, format="ISO8601", errors="coerce")
    retry = parsed.isna() & values.notna()
    if retry.any():
        parsed[retry] = pd.to_datetime(values[retry], dayfirst=True, errors="coerce")
    return ((parsed - EPOCH).dt.days).astype(float).to_numpy()


def build_fields(
    profile: SchemaProfile, calibration: CalibrationStats, sample: pd.DataFrame | None = None
) -> list[Field]:
    """Fields for every learnable profile column, parents before their dependent dates."""
    use_sample = sample is not None and not sample.empty
    parents = {child: parent for (parent, child) in calibration.gaps_days}
    if use_sample:
        for child, parent in temporal_pairs(profile).items():
            parents.setdefault(child, parent)

    fields: dict[str, Field] = {}
    # Two passes: plain columns first, so a dependent date can find its parent field
    for second_pass in (False, True):
        for col in profile.columns:
            if is_identifier(col) or (col.name in parents) != second_pass:
                continue
            observed = sample[col.name] if use_sample and col.name in sample.columns else None
            if observed is not None and observed.notna().sum() < MIN_OBSERVED:
                observed = None
            made = _field(col, calibration, observed, parents.get(col.name), fields, sample)
            if made is not None:
                fields[col.name] = made
    return list(fields.values())


def _missing_share(col: ColumnSpec, calibration: CalibrationStats, observed) -> float:
    if not col.nullable:
        return 0.0
    if observed is not None:
        return float(observed.isna().mean())
    return float(calibration.null_rate.get(col.name, 0.0))


def _field(col, calibration, observed, parent, built, sample) -> Field | None:
    missing = _missing_share(col, calibration, observed)
    base = {"name": col.name, "missing": missing}
    if observed is not None:
        base["observed_missing"] = observed.isna().to_numpy()

    if col.type in {"category", "boolean"} or col.allowed_values or col.name in calibration.categorical:
        return _category_field(col, calibration, observed, base)
    if col.type == "date":
        if parent is not None and parent in built and built[parent].kind in {"date", "gap"}:
            gap = _gap_field(col, calibration, observed, parent, sample, base)
            if gap is not None:
                return gap
        return _date_field(col, calibration, observed, base)
    if col.type in {"integer", "decimal"}:
        return _number_field(col, calibration, observed, base)
    return None  # free text and strings with no statistics: knowledge-based fallback


def _category_field(col, calibration, observed, base) -> Field | None:
    allowed = [str(v) for v in col.allowed_values] if col.allowed_values else None
    if observed is not None:
        values = observed.dropna().astype(str)
        if allowed:
            values = values[values.isin(allowed)]
        shares = values.value_counts(normalize=True)
        if not shares.empty:
            data = np.array(
                [str(v) if pd.notna(v) and str(v) in shares.index else None for v in observed],
                dtype=object,
            )
            return Field(
                **base,
                kind="category",
                categories=[str(c) for c in shares.index],
                probs=shares.to_numpy(dtype=float),
                data=data,
            )
        base.pop("observed_missing", None)

    shares = calibration.categorical.get(col.name, {})
    if allowed:
        shares = {k: v for k, v in shares.items() if k in allowed} or dict.fromkeys(allowed, 1.0)
    if not shares and col.type == "boolean":
        shares = {"True": 0.5, "False": 0.5}
    if not shares:
        return None
    probs = np.array(list(shares.values()), dtype=float)
    return Field(**base, kind="category", categories=list(shares), probs=probs / probs.sum())


def _number_field(col, calibration, observed, base) -> Field | None:
    integer = col.type == "integer"
    decimals = 0 if integer else (col.decimals if col.decimals is not None else 2)
    lo = float(col.min) if isinstance(col.min, int | float) else None
    hi = float(col.max) if isinstance(col.max, int | float) else None
    common = {"kind": "number", "integer": integer, "decimals": decimals}
    if observed is not None:
        data = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
        if (~np.isnan(data)).sum() >= MIN_OBSERVED:
            present = data[~np.isnan(data)]
            return Field(
                **base, **common, data=data,
                lo=lo if lo is not None else float(present.min()),
                hi=hi if hi is not None else float(present.max()),
            )
        base.pop("observed_missing", None)
    spec = calibration.numeric.get(col.name)
    if spec is None:
        return None
    return Field(
        **base, **common, dist=dict(spec),
        lo=lo if lo is not None else spec.get("min"),
        hi=hi if hi is not None else spec.get("max"),
    )


def _date_field(col, calibration, observed, base) -> Field | None:
    if observed is not None:
        data = to_days(observed)
        if (~np.isnan(data)).sum() >= MIN_OBSERVED:
            return Field(**base, kind="date", data=data)
        base.pop("observed_missing", None)
    span = calibration.date_ranges.get(col.name)
    if span is None:
        return None
    start = float((pd.Timestamp(span["start"]) - EPOCH).days)
    end = float((pd.Timestamp(span["end"]) - EPOCH).days)
    return Field(**base, kind="date", dist={"start": start, "end": end}, lo=start, hi=end)


def _gap_field(col, calibration, observed, parent, sample, base) -> Field | None:
    if observed is not None and parent in sample.columns:
        gaps = to_days(observed) - to_days(sample[parent])
        if (~np.isnan(gaps)).sum() >= MIN_OBSERVED:
            return Field(**base, kind="gap", data=gaps, parent=parent)
    spec = calibration.gaps_days.get((parent, col.name))
    if spec is None:
        return None
    base.pop("observed_missing", None)
    return Field(**base, kind="gap", dist=dict(spec), parent=parent)


def field_quantiles(f: Field, qs: list[float]) -> list[float]:
    """Quantiles of a number, date or gap field, from the sample or the calibration."""
    if f.learned_from_sample:
        return [float(v) for v in np.quantile(f.present_values(), qs)]
    d = f.dist
    if f.kind == "date":
        return [d["start"] + q * (d["end"] - d["start"]) for q in qs]
    if f.kind == "gap":
        return [max(d["mean"] + d.get("std", 0) * norm.ppf(q), d.get("min", 1)) for q in qs]
    if d.get("dist") == "lognormal":
        out = [float(np.exp(d["mean"] + d["std"] * norm.ppf(q))) for q in qs]
    elif "mean" in d:
        out = [d["mean"] + d["std"] * norm.ppf(q) for q in qs]
    else:
        out = [f.lo + q * (f.hi - f.lo) for q in qs]
    return [float(np.clip(v, f.lo, f.hi)) for v in out]


def assemble(
    profile: SchemaProfile,
    calibration: CalibrationStats,
    fields: list[Field],
    draws: dict[str, np.ndarray],
    present: dict[str, np.ndarray],
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Turn raw draws into the output table, in the profile's column order.

    ``draws`` holds category names, numbers, day counts or gap days per field; ``present``
    says which rows have a value. Columns without a field are filled by the knowledge-based
    generator.
    """
    by_name = {f.name: f for f in fields}
    days: dict[str, np.ndarray] = {}
    out: dict[str, pd.Series] = {}
    for f in fields:  # parents come before their dependent dates
        raw = draws[f.name]
        keep = present.get(f.name, np.ones(n, dtype=bool))
        if f.kind == "category":
            values = pd.Series(raw, dtype=object)
        elif f.kind == "number":
            values = _format_number(f, np.asarray(raw, dtype=float))
        else:
            d = np.round(np.asarray(raw, dtype=float))
            if f.kind == "gap":
                d = days[f.parent] + np.maximum(d, 1)
            d[~keep] = np.nan
            days[f.name] = d
            values = _format_dates(d)
        out[f.name] = values.where(pd.Series(keep), None)

    fallback = KnowledgeBasedGenerator()
    fallback.fit(profile, calibration)
    rng = np.random.default_rng(seed + 7919)
    for col in profile.columns:
        if col.name not in by_name:
            out[col.name] = fallback.sample_column(col, n, rng)
    return pd.DataFrame({c.name: out[c.name].reset_index(drop=True) for c in profile.columns})


def _format_number(f: Field, values: np.ndarray) -> pd.Series:
    values = np.clip(values, f.lo if f.lo is not None else -np.inf, f.hi if f.hi is not None else np.inf)
    if f.integer:
        return pd.Series(np.round(values).astype(int), dtype=object)
    return pd.Series(np.round(values, f.decimals), dtype=object)


def _format_dates(d: np.ndarray) -> pd.Series:
    stamps = EPOCH + pd.to_timedelta(np.nan_to_num(d), unit="D")
    text = pd.Series(stamps.strftime("%Y-%m-%d"), dtype=object)
    return text.where(pd.Series(~np.isnan(d)), None)
