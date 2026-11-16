"""Realism metrics comparing synthetic data with the reference dataset and held-back statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from synthgen.reference_world.stores import TRIAL_TYPES, VALUE_BANDS, VALUE_LABELS


def ks_distance(synthetic: pd.Series, reference: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic for numeric or date columns (0 = identical)."""
    a, b = _to_numeric(synthetic), _to_numeric(reference)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def js_distance(synthetic: pd.Series, reference: pd.Series) -> float:
    """Jensen-Shannon distance between category distributions (0 = identical, 1 = disjoint)."""
    cats = sorted(set(synthetic.dropna().astype(str)) | set(reference.dropna().astype(str)))
    p = synthetic.dropna().astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    q = reference.dropna().astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    return float(jensenshannon(p.values, q.values, base=2))


def proportion_gap(synthetic: pd.Series, target: dict[str, float]) -> float:
    """Total variation distance between generated proportions and a held-back statistic."""
    observed = synthetic.dropna().astype(str).value_counts(normalize=True)
    keys = set(observed.index) | set(target)
    return 0.5 * sum(abs(observed.get(k, 0.0) - target.get(k, 0.0)) for k in keys)


def column_report(
    synthetic: pd.DataFrame, reference: pd.DataFrame, types: dict[str, str]
) -> pd.DataFrame:
    """One row per shared column with the appropriate distance."""
    rows = []
    for col in synthetic.columns.intersection(reference.columns):
        kind = types.get(col, "string")
        if kind in {"integer", "decimal", "date"}:
            rows.append(
                {
                    "column": col,
                    "metric": "ks",
                    "value": ks_distance(synthetic[col], reference[col]),
                }
            )
        elif kind in {"category", "boolean"}:
            rows.append(
                {
                    "column": col,
                    "metric": "js",
                    "value": js_distance(synthetic[col], reference[col]),
                }
            )
    return pd.DataFrame(rows)


def _to_numeric(s: pd.Series) -> np.ndarray:
    s = s.dropna()
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float)
    dates = pd.to_datetime(s, errors="coerce").dropna()
    return (dates.astype("int64") // 86_400_000_000_000).to_numpy(dtype=float)


# --- Held-back statistics ----------------------------------------------------------------
# Each check recomputes one held-back statistic on the synthetic data (in reference-world
# column names) and returns a gap in 0..1 (0 = matches). A subgroup the synthetic data never
# produces counts as the largest gap, 1.


def _flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _group_share_gap(df: pd.DataFrame, group: str, col: str, targets: dict) -> float:
    gaps = []
    for name, target in targets.items():
        rows = df[df[group].astype(str) == name][col].dropna()
        gaps.append(proportion_gap(rows, target) if len(rows) else 1.0)
    return float(np.mean(gaps)) if gaps else float("nan")


def _quantile_gap(values: pd.Series, target: dict[str, float]) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 1.0
    gaps = []
    for key, ref in target.items():
        syn = float(values.quantile(int(key[1:]) / 100))
        gaps.append(min(abs(syn - ref) / ref, 1.0) if ref else float(syn != ref))
    return float(np.mean(gaps))


def _rate_gaps(df: pd.DataFrame, groups: pd.Series, flag: pd.Series, targets: dict) -> float:
    gaps = []
    for name, target in targets.items():
        mask = groups.astype(str) == name
        gaps.append(abs(float(flag[mask].mean()) - target) if mask.any() else 1.0)
    return float(np.mean(gaps)) if gaps else float("nan")


def _days(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(later, errors="coerce") - pd.to_datetime(earlier, errors="coerce")
    ).dt.days


def _claim_quantiles(df, held):
    targets = held["claim_amount_quantiles_by_claim_type"]
    gaps = [
        _quantile_gap(df[df["claim_type"].astype(str) == name]["claim_amount"], target)
        for name, target in targets.items()
    ]
    return float(np.mean(gaps))


def _defended_by_band(df, held):
    amount = pd.to_numeric(df["claim_amount"], errors="coerce")
    bands = pd.cut(amount, VALUE_BANDS, labels=VALUE_LABELS, right=False)
    return _rate_gaps(df, bands, _flag(df["defended"]), held["defended_rate_by_value_band"])


def _trial_weeks(df, held):
    trials = df[df["hearing_type"].isin(TRIAL_TYPES)]
    weeks = _days(trials["hearing_date"], trials["issue_date"]) / 7
    small = trials["track"] == "Small claims"
    gaps = []
    for name, part in [("small claims", small), ("fast, intermediate and multi", ~small)]:
        ref = held["trial_weeks_median_by_track_group"][name]
        syn = weeks[part].median()
        gaps.append(1.0 if pd.isna(syn) else min(abs(syn - ref) / ref, 1.0))
    return float(np.mean(gaps))


def _warrant_by_value(df, held):
    judged = df[df["judgment_date"].notna()]
    low = pd.to_numeric(judged["claim_amount"], errors="coerce") < 1000
    groups = low.map({True: "under 1000", False: "1000 and over"})
    warrant = judged["warrant_issue_date"].notna()
    return _rate_gaps(judged, groups, warrant, held["warrant_rate_by_value"])


HELD_BACK_CHECKS = {
    "claim_amount_quantiles_by_claim_type": ({"claim_amount", "claim_type"}, _claim_quantiles),
    "track_by_claim_type": (
        {"track", "claim_type"},
        lambda df, held: _group_share_gap(
            df[df["track"].notna()], "claim_type", "track", held["track_by_claim_type"]
        ),
    ),
    "status_by_claim_type": (
        {"case_status", "claim_type"},
        lambda df, held: _group_share_gap(
            df, "claim_type", "case_status", held["status_by_claim_type"]
        ),
    ),
    "outcome_among_heard": (
        {"outcome", "case_status"},
        lambda df, held: proportion_gap(
            df[df["case_status"] == "Heard"]["outcome"], held["outcome_among_heard"]
        ),
    ),
    "defended_rate_by_value_band": ({"defended", "claim_amount"}, _defended_by_band),
    "trial_weeks_median_by_track_group": (
        {"hearing_date", "issue_date", "track", "hearing_type"},
        _trial_weeks,
    ),
    "warrant_rate_by_value": (
        {"warrant_issue_date", "judgment_date", "claim_amount"},
        _warrant_by_value,
    ),
}


def held_back_gaps(synthetic: pd.DataFrame, held: dict) -> dict[str, float]:
    """Gap from every held-back statistic the synthetic columns allow us to recompute."""
    return {
        name: check(synthetic, held)
        for name, (needs, check) in HELD_BACK_CHECKS.items()
        if needs <= set(synthetic.columns) and name in held
    }


# --- Usefulness of patterns: train on synthetic, test on reference -----------------------


class NaiveBayes:
    """Categorical naive Bayes with Laplace smoothing. Numeric and date features are
    binned into quartiles of the training data, with missing values as their own bin.
    Deliberately simple and dependency-free."""

    def fit(self, df: pd.DataFrame, target: str, features: list[str]) -> NaiveBayes:
        self.target, self.features = target, features
        self.edges = {}
        train = df.dropna(subset=[target])
        binned = self._bin(train, fit=True)
        y = train[target].astype(str)
        self.classes = sorted(y.unique())
        self.log_prior = np.log(y.value_counts(normalize=True).reindex(self.classes))
        self.tables = {}
        for f in features:
            counts = pd.crosstab(binned[f], y).reindex(columns=self.classes, fill_value=0)
            self.tables[f] = np.log((counts + 1) / (counts.sum() + len(counts) + 1))
            self.tables[f].loc["__unseen__"] = np.log(1 / (counts.sum() + len(counts) + 1))
        return self

    def _bin(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for f in self.features:
            values = _to_number(df[f])
            if (fit and values.nunique() > 8) or (not fit and f in self.edges):
                if fit:
                    self.edges[f] = np.unique(np.nanquantile(values, [0.25, 0.5, 0.75]))
                bins = pd.Series(np.digitize(values, self.edges[f]), index=df.index).astype(str)
                out[f] = bins.where(values.notna(), "missing")
            else:
                out[f] = df[f].astype(str)
        return out

    def predict(self, df: pd.DataFrame) -> pd.Series:
        binned = self._bin(df)
        scores = np.tile(self.log_prior.to_numpy(), (len(df), 1))
        for f, table in self.tables.items():
            keys = binned[f].where(binned[f].isin(table.index), "__unseen__")
            scores += table.loc[keys].to_numpy()
        return pd.Series(np.array(self.classes)[scores.argmax(axis=1)], index=df.index)


def _to_number(s: pd.Series) -> pd.Series:
    """Numbers as floats and dates as day numbers, whether parsed already or still text."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - pd.Timestamp("2000-01-01")).dt.days.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() >= s.notna().sum() * 0.9 and numeric.notna().any():
        return numeric
    dates = pd.to_datetime(s, errors="coerce", format="ISO8601")
    return (dates - pd.Timestamp("2000-01-01")).dt.days.astype(float)


def tstr_accuracy(
    train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]
) -> float:
    """Accuracy on ``test`` of a model trained on ``train`` (train-on-synthetic, test-on-real)."""
    test = test.dropna(subset=[target])
    if not features or test.empty or train[target].dropna().empty:
        return float("nan")
    model = NaiveBayes().fit(train, target, features)
    return float((model.predict(test) == test[target].astype(str)).mean())
