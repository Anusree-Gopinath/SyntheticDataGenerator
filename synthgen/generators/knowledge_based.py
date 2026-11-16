"""Knowledge-based generator (the baseline).

Uses only what people have written down: the approved profile and its rules, and the
calibration statistics. Nothing is learned from example rows.

- Each column is drawn from its calibration statistics, kept to the profile's allowed values.
- Conditional distributions: where the calibration store gives shares within groups (for
  example the defended rate for each claim type), the column is drawn group by group.
- Dependencies come from the approved rules (see ``dependencies.py``): a later date is drawn
  as its earlier date plus a gap, a number is capped at the one it mustn't exceed, and
  conditional presence and value rules are applied in the rows they cover. Columns are
  generated parents first.

Rules it can't read are still enforced by the validator, which rejects rows that break them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthgen.core.models import ColumnSpec, SchemaProfile
from synthgen.generators.base import CalibrationStats
from synthgen.generators.dependencies import DependencyPlan, Order, build_plan
from synthgen.rules.expressions import compile_expression

# Gap used for a date order rule when the calibration store has no gap for that pair
DEFAULT_GAP = {"mean": 60.0, "std": 30.0}


class KnowledgeBasedGenerator:
    name = "knowledge_based"
    supports_scenarios = frozenset({1, 2})

    def __init__(self) -> None:
        self.profile: SchemaProfile | None = None
        self.calibration = CalibrationStats()
        self.plan: DependencyPlan | None = None
        self.conditional: dict[tuple[str, str], dict[str, dict[str, float]]] = {}

    def fit(self, profile, calibration, sample=None) -> None:
        self.profile = profile
        self.calibration = calibration
        self.conditional = conditional_shares(calibration)
        names = {c.name for c in profile.columns}
        edges = [e for e in [*calibration.gaps_days, *self.conditional] if set(e) <= names]
        self.plan = build_plan(profile, edges)

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.profile is None or self.plan is None:
            raise RuntimeError("Call fit() before sample()")
        rng = np.random.default_rng(seed)
        data: dict[str, pd.Series] = {}
        for name in self.plan.order:
            col = self.profile.column(name)
            values = self._apply_nulls(col, self._draw(col, n, rng, data), rng)
            values = self._orders(col, values, rng, data)
            values = self._presence(col, values, n, rng, data)
            data[name] = self._limit(col, values, rng, data)
        return pd.DataFrame({c.name: data[c.name] for c in self.profile.columns})

    def sample_column(self, col: ColumnSpec, n: int, rng: np.random.Generator) -> pd.Series:
        """One independent column with its nulls. The other approaches use this for columns
        they have nothing to learn from (identifiers, columns with no statistics)."""
        return self._apply_nulls(col, self._column(col, n, rng), rng)

    # -- one column, given the columns already made ----------------------------------------

    def _draw(self, col: ColumnSpec, n: int, rng, data: dict[str, pd.Series]) -> pd.Series:
        gap_from = self._gap_parent(col.name)
        if gap_from is not None and gap_from[0] in data:
            return self._dependent_date(data[gap_from[0]], gap_from[1], rng)
        for (parent, child), shares in self.conditional.items():
            if child == col.name and parent in data:
                return self._conditional(col, data[parent], shares, rng)
        return self._column(col, n, rng)

    def _gap_parent(self, name: str) -> tuple[str, dict] | None:
        """The date this column follows and the gap to use, from the rules or calibration."""
        order: Order | None = self.plan.date_parent(name)
        gaps = self.calibration.gaps_days
        if order is not None:
            gap = dict(gaps.get((order.low, name), DEFAULT_GAP))
            gap["min"] = max(gap.get("min", 0), 1 if order.strict else 0)
            return order.low, gap
        for (parent, child), gap in gaps.items():
            if child == name:
                return parent, gap
        return None

    def _conditional(self, col: ColumnSpec, parent: pd.Series, shares: dict, rng) -> pd.Series:
        """Draw within each parent group; rows with an unknown or empty parent use the
        column's overall shares."""
        out = self._column(col, len(parent), rng).astype(object)
        for value, within in shares.items():
            rows = (parent.astype(str) == str(value)).to_numpy()
            weights = self._allowed_shares(col, within)
            if rows.any() and weights:
                values, probs = zip(*weights.items())
                out[rows] = rng.choice(values, size=int(rows.sum()), p=probs)
        return out

    def _orders(self, col: ColumnSpec, values: pd.Series, rng, data) -> pd.Series:
        """Order rules this column must meet (all rows, or the rows an ``or`` rule leaves).

        Numbers: kept at or below the column they mustn't exceed. Dates with a condition:
        redrawn as their earlier date plus a gap (dates without one are drawn that way).
        """
        n = len(values)
        for order in self.plan.orders:
            if order.target != col.name:
                continue
            other = order.high if order.kind == "number" else order.low
            if other not in data:
                continue
            rows = self._where(order.when, {**data, col.name: values}, n)
            if rows is None:
                continue
            if order.kind == "date":
                if order.when is None:
                    continue  # already drawn from its earlier date
                gap = dict(self.calibration.gaps_days.get((order.low, col.name), DEFAULT_GAP))
                gap["min"] = max(gap.get("min", 0), 1 if order.strict else 0)
                later = pd.to_datetime(values, errors="coerce")
                earlier = pd.to_datetime(data[order.low], errors="coerce")
                broken = later <= earlier if order.strict else later < earlier
                bad = (rows & earlier.notna() & (broken | later.isna())).to_numpy()
                if bad.any():
                    fresh = self._dependent_date(data[order.low], gap, rng)
                    values = values.astype(object).copy()
                    values[bad] = fresh[bad]
                continue
            step = 0.0
            if order.strict:
                step = 1.0 if col.type == "integer" else 10.0 ** -(col.decimals or 2)
            limit = pd.to_numeric(data[order.high], errors="coerce") - step
            limit = np.floor(limit) if col.type == "integer" else limit.round(col.decimals or 2)
            numeric = pd.to_numeric(values, errors="coerce")
            over = ((numeric > limit) | (numeric.isna() & limit.notna() & (order.when is not None)))
            over = (over & rows).fillna(False)
            values = values.astype(object).where(~over, limit.where(over))
        return values

    def _presence(self, col: ColumnSpec, values: pd.Series, n: int, rng, data) -> pd.Series:
        """Conditional presence rules: fill or empty this column where the condition holds."""
        for rule in self.plan.presence:
            if rule.column != col.name:
                continue
            rows = self._where(rule.when, {**data, col.name: values}, n)
            if rows is None:
                continue
            if rule.present:
                missing = rows & values.isna()
                if missing.any():
                    fresh = self._draw(col, n, rng, data).astype(object)
                    values = values.astype(object).where(~missing, fresh)
            else:
                values = values.astype(object).where(~rows, None)
        return values

    def _limit(self, col: ColumnSpec, values: pd.Series, rng, data) -> pd.Series:
        """Conditional value and bound rules: redraw this column where its value isn't allowed."""
        n = len(values)
        for rule in self.plan.limits:
            if rule.column != col.name:
                continue
            rows = self._where(rule.when, {**data, col.name: values}, n)
            if rows is None:
                continue
            if rule.low is not None or rule.high is not None:
                values = self._within_bounds(col, values, rows, rule, rng)
                continue
            ok = values.isin(rule.allowed) if rule.allowed is not None else ~values.isin(rule.excluded)
            bad = (rows & ~ok).to_numpy()
            if not bad.any():
                continue
            if rule.allowed is not None:
                shares = self.calibration.categorical.get(col.name, {})
                options = {v: shares.get(str(v), 0.0) for v in rule.allowed}
                if not any(options.values()):
                    options = {v: 1.0 for v in rule.allowed}
            else:
                options = {
                    v: p for v, p in self._overall_shares(col).items() if v not in rule.excluded
                }
            total = sum(options.values())
            if not options or total <= 0:
                continue
            choices, probs = list(options), np.array(list(options.values())) / total
            values = values.astype(object).copy()
            values[bad] = rng.choice(np.array(choices, dtype=object), size=int(bad.sum()), p=probs)
        return values

    def _within_bounds(self, col: ColumnSpec, values: pd.Series, rows: pd.Series, rule, rng):
        """Redraw, evenly between the bounds, the rows whose date or number is outside them.
        A missing bound is taken from the column's usual range."""
        is_date = col.type == "date"
        as_num = (
            (lambda v: pd.to_datetime(v, errors="coerce"))
            if is_date
            else (lambda v: pd.to_numeric(v, errors="coerce"))
        )
        if is_date:
            usual = self.calibration.date_ranges.get(col.name, {"start": "2024-01-01", "end": "2025-12-31"})
            lo = pd.Timestamp(rule.low if rule.low is not None else usual["start"])
            hi = pd.Timestamp(rule.high if rule.high is not None else usual["end"])
            step = pd.Timedelta(days=1)
        else:
            stats = self.calibration.numeric.get(col.name, {})
            lo = float(rule.low if rule.low is not None else stats.get("min", col.min or 0))
            hi = float(rule.high if rule.high is not None else stats.get("max", col.max or lo + 1000))
            step = 1.0 if col.type == "integer" else 10.0 ** -(col.decimals or 2)
        lo = lo + step if rule.strict_low else lo
        hi = hi - step if rule.strict_high else hi
        if lo > hi:
            return values  # nothing fits; the validator reports the rule
        current = as_num(values)
        outside = rows & values.notna() & ~((current >= lo) & (current <= hi)).fillna(False)
        bad = outside.to_numpy()
        if not bad.any():
            return values
        values = values.astype(object).copy()
        k = int(bad.sum())
        if is_date:
            offsets = rng.integers(0, (hi - lo).days + 1, size=k)
            values[bad] = (lo + pd.to_timedelta(offsets, unit="D")).strftime("%Y-%m-%d").to_numpy()
        elif col.type == "integer":
            values[bad] = rng.integers(int(np.ceil(lo)), int(np.floor(hi)) + 1, size=k)
        else:
            values[bad] = np.round(rng.uniform(lo, hi, size=k), col.decimals or 2)
        return values

    def _where(self, when: str | None, data: dict[str, pd.Series], n: int) -> pd.Series | None:
        if not when:
            return pd.Series(True, index=range(n))
        try:
            return compile_expression(when)(pd.DataFrame(data, index=range(n)))
        except (KeyError, TypeError):  # can't be judged yet; leave it to the validator
            return None

    # -- column samplers ---------------------------------------------------------------

    @property
    def _unique(self) -> set[str]:
        return self.plan.unique if self.plan is not None else set()

    def _allowed_shares(self, col: ColumnSpec, shares: dict[str, float]) -> dict[str, float]:
        """Shares kept to the profile's allowed values, scaled to add up to 1."""
        if col.allowed_values:
            allowed = {str(v) for v in col.allowed_values}
            shares = {k: v for k, v in shares.items() if str(k) in allowed}
        total = sum(shares.values())
        return {k: v / total for k, v in shares.items()} if total > 0 else {}

    def _overall_shares(self, col: ColumnSpec) -> dict[str, float]:
        shares = self._allowed_shares(col, self.calibration.categorical.get(col.name, {}))
        if not shares and col.allowed_values:
            shares = {v: 1 / len(col.allowed_values) for v in col.allowed_values}
        return shares

    def _column(self, col: ColumnSpec, n: int, rng: np.random.Generator) -> pd.Series:
        shares = self._overall_shares(col)
        if shares:
            values, probs = zip(*shares.items())
            return pd.Series(rng.choice(values, size=n, p=np.array(probs) / sum(probs)))
        if col.type in {"integer", "decimal"}:
            return self._numeric(col, n, rng)
        if col.type == "date":
            return self._date(col, n, rng)
        if col.concept == "case_reference" or col.unique or col.name in self._unique:
            return self._case_numbers(n, rng)
        return pd.Series([f"{col.name}_{i}" for i in range(n)])

    def _numeric(self, col: ColumnSpec, n: int, rng: np.random.Generator) -> pd.Series:
        stats = self.calibration.numeric.get(col.name, {})
        lo = float(stats.get("min", col.min if col.min is not None else 0))
        hi = float(stats.get("max", col.max if col.max is not None else 1000))
        if stats.get("dist") == "lognormal":
            values = rng.lognormal(stats["mean"], stats["std"], size=n)
        elif "mean" in stats:
            values = rng.normal(stats["mean"], stats["std"], size=n)
        else:
            values = rng.uniform(lo, hi, size=n)
        values = np.clip(values, lo, hi)
        if col.type == "integer":
            return pd.Series(np.round(values).astype(int))
        return pd.Series(np.round(values, col.decimals if col.decimals is not None else 2))

    def _date(self, col: ColumnSpec, n: int, rng: np.random.Generator) -> pd.Series:
        rng_cfg = self.calibration.date_ranges.get(col.name, {"start": "2024-01-01", "end": "2025-12-31"})
        start, end = pd.Timestamp(rng_cfg["start"]), pd.Timestamp(rng_cfg["end"])
        offsets = rng.integers(0, (end - start).days + 1, size=n)
        return pd.Series((start + pd.to_timedelta(offsets, unit="D")).strftime("%Y-%m-%d"))

    def _dependent_date(self, parent: pd.Series, gap: dict, rng: np.random.Generator) -> pd.Series:
        days = rng.normal(gap["mean"], gap.get("std", 0), size=len(parent))
        days = np.maximum(days, gap.get("min", 1)).round().astype(int)
        base = pd.to_datetime(parent, errors="coerce")
        out = (base + pd.to_timedelta(days, unit="D")).dt.strftime("%Y-%m-%d")
        return out.where(base.notna(), None)

    def _case_numbers(self, n: int, rng: np.random.Generator) -> pd.Series:
        letters = np.array(list("ABCDEFGHJKLMNPRSTUVWXYZ"))
        seen: set[str] = set()
        out = []
        while len(out) < n:
            ref = (
                f"K{rng.integers(0, 100):02d}"
                f"{''.join(rng.choice(letters, 2))}{rng.integers(0, 1000):03d}"
            )
            if ref not in seen:
                seen.add(ref)
                out.append(ref)
        return pd.Series(out)

    def _apply_nulls(self, col: ColumnSpec, s: pd.Series, rng: np.random.Generator) -> pd.Series:
        rate = self.calibration.null_rate.get(col.name, 0.0)
        if not col.nullable or rate <= 0:
            return s
        mask = rng.random(len(s)) < rate
        return s.astype(object).where(~mask, None)


def conditional_shares(cal: CalibrationStats) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Shares within groups: the store's ``conditional`` entries, plus each pairwise yes/no
    rate written as True/False shares per group."""
    out = {key: dict(value) for key, value in cal.conditional.items()}
    for key, rates in cal.pairwise.items():
        out.setdefault(
            key, {g: {"True": r, "False": round(1 - r, 6)} for g, r in rates.items()}
        )
    return out
