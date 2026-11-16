"""Evaluate one generated dataset: rule compliance, realism and (Scenario 2) privacy.

Synthetic columns are renamed to the reference world's names by concept, so they can be
compared with the evaluation store. Columns with no reference counterpart are not scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from synthgen.core.models import SchemaProfile
from synthgen.evaluation.privacy import (
    density_membership_attack,
    distance_membership_attack,
    distance_to_closest_record,
    privacy_threshold,
)
from synthgen.evaluation.realism import column_report, held_back_gaps, tstr_accuracy
from synthgen.reference_world.stores import REFERENCE_CONCEPTS, profile_column_map
from synthgen.rules.validator import validate

# Preferred outcome to predict in the train-on-synthetic test, first one present wins
TSTR_TARGETS = ["case_status", "track", "outcome", "hearing_type", "claim_type"]
# Identifiers are unique by design, so they say nothing about realism or privacy
IDENTIFIERS = {"case_number"}


@dataclass
class EvaluationStores:
    """What the evaluators may read. Never passed to generators."""

    reference: pd.DataFrame
    held_back: dict
    sample: pd.DataFrame | None = None  # Scenario 2 only
    candidates: pd.DataFrame | None = None  # Scenario 2 only
    privacy_percentile: float = 5


def to_reference_names(data: pd.DataFrame, profile: SchemaProfile) -> pd.DataFrame:
    """Keep the columns that match a reference column, renamed to the reference names."""
    to_ref = {col: ref for ref, col in profile_column_map(profile).items()}
    return data[[c for c in data.columns if c in to_ref]].rename(columns=to_ref)


def reference_types(profile: SchemaProfile) -> dict[str, str]:
    to_ref = {col: ref for ref, col in profile_column_map(profile).items()}
    return {to_ref[c.name]: c.type for c in profile.columns if c.name in to_ref}


def tstr_setup(columns: list[str]) -> tuple[str | None, list[str]]:
    target = next((t for t in TSTR_TARGETS if t in columns), None)
    features = [c for c in columns if c != target and c not in IDENTIFIERS]
    return target, features


def reference_baseline(stores: EvaluationStores, profile: SchemaProfile, seed: int = 0):
    """Train-on-reference accuracy for the same task, for comparison with each approach.

    Uses the same held-out half of the reference world that the synthetic models are
    tested on.
    """
    cols = list(reference_types(profile))
    target, features = tstr_setup(cols)
    if target is None:
        return None, None, float("nan")
    test = stores.reference.sample(frac=0.5, random_state=seed)
    train = stores.reference.drop(test.index)
    return target, test, tstr_accuracy(train, test, target, features)


def _privacy_frame(df: pd.DataFrame, types: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Dates become day numbers so Gower distance treats them as numeric."""
    out = df[[c for c in df.columns if c not in IDENTIFIERS]].copy()
    numeric = []
    for col in out.columns:
        if types.get(col) in {"integer", "decimal"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            numeric.append(col)
        elif types.get(col) == "date":
            dates = pd.to_datetime(out[col], errors="coerce", format="ISO8601")
            out[col] = (dates - pd.Timestamp("2000-01-01")).dt.days.astype(float)
            numeric.append(col)
        else:
            out[col] = out[col].astype(str)
    return out, numeric


def evaluate_run(
    data: pd.DataFrame,
    profile: SchemaProfile,
    stores: EvaluationStores,
    tstr_test: pd.DataFrame | None = None,
) -> dict[str, float]:
    """All metrics for one run. ``tstr_test`` is the reference half from reference_baseline."""
    result = validate(data, profile)
    metrics: dict[str, float] = {
        "rows_delivered": len(data),
        "rule_compliance": float(result.passed.mean()) if len(data) else float("nan"),
    }

    synthetic = to_reference_names(data, profile)
    types = reference_types(profile)
    report = column_report(synthetic, stores.reference, types)
    metrics["reference_distance"] = float(report["value"].mean()) if len(report) else float("nan")
    for row in report.itertuples():
        metrics[f"distance:{row.column}"] = row.value
    gaps = held_back_gaps(synthetic, stores.held_back)
    metrics["held_back_gap"] = float(np.mean(list(gaps.values()))) if gaps else float("nan")
    metrics["held_back_checks"] = len(gaps)

    target, features = tstr_setup(list(synthetic.columns))
    metrics["tstr_accuracy"] = (
        tstr_accuracy(synthetic, tstr_test, target, features)
        if target and tstr_test is not None
        else float("nan")
    )

    if stores.sample is not None and stores.candidates is not None:
        cols = [c for c in synthetic.columns if c in REFERENCE_CONCEPTS]
        syn, numeric = _privacy_frame(synthetic[cols], types)
        sample, _ = _privacy_frame(stores.sample[cols], types)
        pool, _ = _privacy_frame(stores.candidates[cols], types)
        pool["member"] = stores.candidates["member"].to_numpy()
        reference, _ = _privacy_frame(stores.reference[cols], types)

        dcr = distance_to_closest_record(syn, sample, numeric)
        threshold = privacy_threshold(sample, numeric, stores.privacy_percentile)
        distance = distance_membership_attack(syn, pool, numeric)
        density = density_membership_attack(syn, pool, reference, numeric)
        metrics.update(
            {
                "dcr_median": float(np.median(dcr)),
                "near_copy_share": float((dcr < threshold).mean()),
                "exact_copies": int((dcr == 0).sum()),
                "privacy_threshold": threshold,
                "distance_attack_auc": distance["auc"],
                "distance_attack_accuracy": distance["accuracy"],
                "density_attack_auc": density["auc"],
            }
        )
    return metrics
