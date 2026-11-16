"""Privacy evaluation (Scenario 2 only), under the threat model in the proposal.

Threat model:
- Goal: decide whether a candidate row was in the uploaded sample.
- Access: the generated dataset and the schema profile only.
- Knowledge: a candidate pool from the reference world, half of which were in the sample.

Results are evidence about these specific risks under these test conditions,
not proof that the data is private.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def gower_distances(a: pd.DataFrame, b: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    """Gower distance matrix between rows of ``a`` and ``b`` (values in 0..1).

    Numeric columns: absolute difference scaled by the combined range.
    Other columns: 0 if equal, 1 otherwise.
    Missing values: 0 if both rows are missing the value, 1 if only one is.
    """
    cols = list(a.columns.intersection(b.columns))
    total = np.zeros((len(a), len(b)))
    for col in cols:
        if col in numeric_cols:
            x = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
            both = np.concatenate([x, y])
            span = np.nanmax(both) - np.nanmin(both) if np.isfinite(both).any() else 0.0
            diff = np.abs(x[:, None] - y[None, :])
            d = np.where(np.isnan(diff), 1.0, diff / span if span > 0 else 0.0)
            d[np.isnan(x)[:, None] & np.isnan(y)[None, :]] = 0.0
        else:
            # fillna first: pandas keeps missing values missing through astype(str)
            x = a[col].astype(object).fillna("<missing>").astype(str).to_numpy()
            y = b[col].astype(object).fillna("<missing>").astype(str).to_numpy()
            d = (x[:, None] != y[None, :]).astype(float)
        total += d
    return total / max(len(cols), 1)


def distance_to_closest_record(
    synthetic: pd.DataFrame, sample: pd.DataFrame, numeric_cols: list[str]
) -> np.ndarray:
    """For each synthetic row, the Gower distance to its nearest sample row."""
    return gower_distances(synthetic, sample, numeric_cols).min(axis=1)


def privacy_threshold(
    sample: pd.DataFrame, numeric_cols: list[str], percentile: float = 5
) -> float:
    """Threshold for 'too close': a low percentile of distances between different sample rows."""
    d = gower_distances(sample, sample, numeric_cols)
    off_diagonal = d[~np.eye(len(sample), dtype=bool)]
    return float(np.percentile(off_diagonal, percentile))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve from ranks (Mann-Whitney U). Higher score = 'member'."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _attack_result(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AUC, plus accuracy when the attacker calls the top half of scores members."""
    guess = scores > np.median(scores)
    return {"auc": roc_auc(labels, scores), "accuracy": float((guess == labels).mean())}


def distance_membership_attack(synthetic, candidates, numeric_cols):
    """Score each candidate by closeness to its nearest synthetic row: closer = member."""
    labels = candidates["member"].astype(str).str.lower().eq("true").to_numpy()
    features = candidates.drop(columns=["member"])
    nearest = gower_distances(features, synthetic, numeric_cols).min(axis=1)
    return _attack_result(labels, -nearest)


def density_membership_attack(synthetic, candidates, reference, numeric_cols, k: int = 5):
    """Compare how dense the synthetic data is around each candidate with how dense the
    reference world is there. Unusually high synthetic density suggests membership."""
    labels = candidates["member"].astype(str).str.lower().eq("true").to_numpy()
    features = candidates.drop(columns=["member"])
    syn = np.sort(gower_distances(features, synthetic, numeric_cols), axis=1)[:, k - 1]
    # The candidates are themselves reference rows, so skip the zero distance to self
    ref = np.sort(gower_distances(features, reference, numeric_cols), axis=1)[:, k]
    return _attack_result(labels, ref / np.maximum(syn, 1e-9))
