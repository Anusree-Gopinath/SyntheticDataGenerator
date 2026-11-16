"""Build the separated stores from the reference dataset.

- Calibration store: aggregate statistics generators are allowed to see (mostly single-column
  summaries, plus a few simple pairwise ones).
- Evaluation store: held-back statistics (subgroup and multi-column patterns) that are never
  given to generators, used to test what they inferred.
- Scenario 2 sample and candidate pool: for small-sample generation and privacy tests.

Run: ``python -m synthgen.reference_world.stores``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from synthgen.core.models import SchemaProfile
from synthgen.generators.base import CalibrationStats

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
VALUE_BANDS = [0, 500, 2000, 10000, 25000, 100000, np.inf]
VALUE_LABELS = ["under 500", "500-2k", "2k-10k", "10k-25k", "25k-100k", "over 100k"]
TRIAL_TYPES = ["Small claims hearing", "Trial"]


def _shares(s: pd.Series) -> dict[str, float]:
    return {str(k): round(float(v), 4) for k, v in s.value_counts(normalize=True).items()}


def _rate(s: pd.Series) -> float:
    return round(float(s.mean()), 4)


def _days(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (pd.to_datetime(later) - pd.to_datetime(earlier)).dt.days


def _quantiles(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    return {f"p{int(q * 100):02d}": round(float(s.quantile(q)), 2) for q in QUANTILES}


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.lower().isin(["true", "1"])


def calibration_statistics(df: pd.DataFrame) -> dict:
    """Summaries generators may see."""
    money = df[df["claim_amount"].notna()]
    trials = df[df["hearing_type"].isin(TRIAL_TYPES)]
    log_amount = np.log(money["claim_amount"])
    return {
        "description": "Calibration statistics: the only summaries generators may see.",
        "n_cases": len(df),
        "issue_date_range": {"start": df["issue_date"].min(), "end": df["issue_date"].max()},
        "categorical": {
            col: _shares(df[col].dropna())
            for col in ["claim_type", "track", "court", "case_status", "hearing_type", "judgment_type"]
        },
        "rates": {
            col: _rate(_bool(df, col))
            for col in ["defended", "claimant_represented", "defendant_represented", "interpreter_required"]
        },
        "claim_amount": {
            **_quantiles(money["claim_amount"]),
            "share_under_500": _rate(money["claim_amount"] < 500),
            "log_mean": round(float(log_amount.mean()), 4),
            "log_std": round(float(log_amount.std()), 4),
        },
        "null_rate": {col: round(float(df[col].isna().mean()), 4) for col in df.columns},
        "days": {
            "issue_to_trial_median_all_tracks": float(_days(trials["hearing_date"], trials["issue_date"]).median()),
            "issue_to_judgment_median": float(_days(df["judgment_date"], df["issue_date"]).median()),
            "judgment_to_warrant_median": float(_days(df["warrant_issue_date"], df["judgment_date"]).median()),
        },
        "pairwise": {
            "defended_rate_by_claim_type": {
                k: _rate(_bool(g, "defended")) for k, g in df.groupby("claim_type")
            },
        },
    }


def held_back_statistics(df: pd.DataFrame) -> dict:
    """Subgroup and multi-column patterns kept for evaluation only."""
    money = df[df["claim_amount"].notna()].copy()
    money["value_band"] = pd.cut(money["claim_amount"], VALUE_BANDS, labels=VALUE_LABELS, right=False)
    allocated = df[df["track"].notna()].copy()
    trials = df[df["hearing_type"].isin(TRIAL_TYPES)].copy()
    trials["days"] = _days(trials["hearing_date"], trials["issue_date"])
    trials["neither_represented"] = ~_bool(trials, "claimant_represented") & ~_bool(trials, "defendant_represented")
    defended = allocated.copy()
    defended["combo"] = np.select(
        [
            _bool(defended, "claimant_represented") & _bool(defended, "defendant_represented"),
            _bool(defended, "claimant_represented"),
            _bool(defended, "defendant_represented"),
        ],
        ["both", "claimant only", "defendant only"],
        default="neither",
    )
    judged = df[df["judgment_date"].notna()].copy()
    judged["warrant"] = judged["warrant_issue_date"].notna()
    judged["low_value"] = judged["claim_amount"] < 1000
    heard = df[df["case_status"] == "Heard"]

    return {
        "description": "Held-back statistics: evaluation only, never given to generators.",
        "claim_amount_quantiles_by_claim_type": {
            k: _quantiles(g["claim_amount"]) for k, g in money.groupby("claim_type")
        },
        "defended_rate_by_value_band": {
            str(k): _rate(_bool(g, "defended")) for k, g in money.groupby("value_band", observed=True)
        },
        "track_by_claim_type": {k: _shares(g["track"]) for k, g in allocated.groupby("claim_type")},
        "trial_weeks_median_by_track_group": {
            "small claims": round(float(trials[trials["track"] == "Small claims"]["days"].median() / 7), 1),
            "fast, intermediate and multi": round(float(trials[trials["track"] != "Small claims"]["days"].median() / 7), 1),
        },
        "trial_weeks_median_by_representation": {
            "neither represented": round(float(trials[trials["neither_represented"]]["days"].median() / 7), 1),
            "at least one represented": round(float(trials[~trials["neither_represented"]]["days"].median() / 7), 1),
        },
        "trial_weeks_median_by_interpreter": {
            str(k): round(float(g["days"].median() / 7), 1)
            for k, g in trials.groupby(_bool(trials, "interpreter_required"))
        },
        "trial_weeks_median_by_court": {
            k: round(float(g["days"].median() / 7), 1) for k, g in trials.groupby("court")
        },
        "representation_among_defended": _shares(defended["combo"]),
        "trial_rate_by_representation": {
            k: _rate(g["hearing_type"].isin(TRIAL_TYPES)) for k, g in defended.groupby("combo")
        },
        "outcome_among_heard": _shares(heard["outcome"].dropna()),
        "status_by_claim_type": {k: _shares(g["case_status"]) for k, g in df.groupby("claim_type")},
        "warrant_rate_by_value": {
            "under 1000": _rate(judged[judged["low_value"]]["warrant"]),
            "1000 and over": _rate(judged[~judged["low_value"] & judged["claim_amount"].notna()]["warrant"]),
        },
        "warrant_rate_by_defendant_represented": {
            str(k): _rate(g["warrant"]) for k, g in judged.groupby(_bool(judged, "defendant_represented"))
        },
        "track_defended_claimant_represented": {
            f"{t} | claimant represented={r}": round(float(len(g) / len(allocated)), 4)
            for (t, r), g in allocated.groupby(["track", _bool(allocated, "claimant_represented")])
        },
        "spearman_claim_amount_vs_trial_days": round(
            float(trials[["claim_amount", "days"]].corr(method="spearman").iloc[0, 1]), 4
        ),
    }


def split_statistics(reference: pd.DataFrame) -> tuple[dict, dict]:
    return calibration_statistics(reference), held_back_statistics(reference)


def to_calibration_stats(cal: dict) -> CalibrationStats:
    """Convert the calibration store into the structure generators use (reference-world columns)."""
    amt = cal["claim_amount"]
    return CalibrationStats(
        categorical={k: v for k, v in cal["categorical"].items() if v},
        numeric={"claim_amount": {"mean": amt["log_mean"], "std": amt["log_std"], "min": 25,
                                  "max": 500000, "dist": "lognormal"}},
        date_ranges={"issue_date": cal["issue_date_range"]},
        gaps_days={("issue_date", "hearing_date"): {
            "mean": cal["days"]["issue_to_trial_median_all_tracks"], "std": 90, "min": 14}},
        null_rate=cal["null_rate"],
        pairwise=(
            {("claim_type", "defended"): cal["pairwise"]["defended_rate_by_claim_type"]}
            if "defended_rate_by_claim_type" in cal.get("pairwise", {})
            else {}
        ),
    )


# Concept of each reference-world column, so statistics can be matched to a profile's columns.
# judgment_type has no dictionary concept, so it is never matched.
REFERENCE_CONCEPTS = {
    "case_number": "case_reference",
    "claim_type": "claim_type",
    "claim_amount": "claim_amount",
    "track": "track",
    "issue_date": "issue_date",
    "court": "court_name",
    "region": "region",
    "defended": "defended_flag",
    "claimant_represented": "claimant_represented",
    "defendant_represented": "defendant_represented",
    "interpreter_required": "interpreter_required",
    "hearing_date": "hearing_date",
    "hearing_type": "hearing_type",
    "case_status": "case_status",
    "outcome": "outcome",
    "judgment_date": "judgment_date",
    "judgment_amount": "judgment_amount",
    "warrant_issue_date": "warrant_issue_date",
}


def profile_column_map(profile: SchemaProfile) -> dict[str, str]:
    """Reference column -> profile column, matched by concept (first profile column wins)."""
    by_concept: dict[str, str] = {}
    for col in profile.columns:
        if col.concept and col.concept not in by_concept:
            by_concept[col.concept] = col.name
    return {
        ref: by_concept[concept]
        for ref, concept in REFERENCE_CONCEPTS.items()
        if concept in by_concept
    }


def calibration_for_profile(cal: dict, profile: SchemaProfile) -> CalibrationStats:
    """Calibration statistics renamed to the profile's column names.

    Statistics for reference columns the profile doesn't have are dropped. Yes/no rates
    become two-value category shares.
    """
    stats = to_calibration_stats(cal)
    names = profile_column_map(profile)
    categorical = {names[k]: v for k, v in stats.categorical.items() if k in names}
    for ref, rate in cal.get("rates", {}).items():
        if ref in names:
            categorical[names[ref]] = {"True": rate, "False": round(1 - rate, 4)}
    return CalibrationStats(
        categorical=categorical,
        numeric={names[k]: v for k, v in stats.numeric.items() if k in names},
        date_ranges={names[k]: v for k, v in stats.date_ranges.items() if k in names},
        gaps_days={
            (names[a], names[b]): v
            for (a, b), v in stats.gaps_days.items()
            if a in names and b in names
        },
        null_rate={names[k]: v for k, v in stats.null_rate.items() if k in names},
        pairwise={
            (names[a], names[b]): v
            for (a, b), v in stats.pairwise.items()
            if a in names and b in names
        },
    )


def sample_for_profile(sample: pd.DataFrame, profile: SchemaProfile) -> pd.DataFrame:
    """The Scenario 2 sample with the profile's column names and order.

    Profile columns with no reference-world counterpart are left empty.
    """
    renamed = sample.rename(columns=profile_column_map(profile))
    return renamed.reindex(columns=[c.name for c in profile.columns])


def draw_sample(reference: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return reference.sample(n=n, random_state=seed).reset_index(drop=True)


def build_candidate_pool(reference: pd.DataFrame, sample: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Half members (from the sample), half non-members (from the rest), labelled."""
    half = n // 2
    members = sample.sample(n=min(half, len(sample)), random_state=seed).assign(member=True)
    rest = reference[~reference["case_number"].isin(sample["case_number"])]
    non_members = rest.sample(n=half, random_state=seed).assign(member=False)
    return pd.concat([members, non_members]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/reference/reference.csv")
    parser.add_argument("--config", default="config/reference_world.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ref = pd.read_csv(args.reference)
    cal, held = split_statistics(ref)
    Path("data/calibration").mkdir(parents=True, exist_ok=True)
    Path("data/evaluation").mkdir(parents=True, exist_ok=True)
    Path("data/calibration/calibration_stats.yaml").write_text(yaml.safe_dump(cal, sort_keys=False, allow_unicode=True))
    Path("data/evaluation/held_back_stats.yaml").write_text(yaml.safe_dump(held, sort_keys=False, allow_unicode=True))
    sample = draw_sample(ref, cfg["stores"]["scenario2_sample_rows"], cfg["seed"])
    sample.to_csv("data/calibration/scenario2_sample.csv", index=False)
    build_candidate_pool(ref, sample, cfg["stores"]["candidate_pool_rows"], cfg["seed"]).to_csv(
        "data/evaluation/candidate_pool.csv", index=False)
    print("Wrote calibration store, held-back statistics, Scenario 2 sample and candidate pool")


if __name__ == "__main__":
    main()
