"""Comparative quality report across approaches for one scenario.

Each figure is the mean and 95% confidence interval across seeds. Differences between
approaches are tested with paired t-tests across seeds (runs with the same seed form a
pair), with the Holm correction for the number of comparisons per metric.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NOT_METRICS = ("approach", "seed", "run_id")
ALPHA = 0.05


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and confidence interval using the t distribution."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf((1 + confidence) / 2, len(arr) - 1) * stats.sem(arr))
    return mean, mean - half, mean + half


def build_report(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI per approach and metric.

    ``metrics`` has one row per run: an ``approach`` column plus one column per metric
    (``seed`` and ``run_id`` columns, if present, are not metrics). Returns one row per
    (approach, metric) with ``mean``, ``lo``, ``hi`` and ``n`` (runs with a value).
    Approaches keep the order they first appear in.
    """
    rows = []
    names = metrics.columns.drop([c for c in NOT_METRICS if c in metrics.columns])
    for approach, runs in metrics.groupby("approach", sort=False):
        for metric in names:
            values = pd.to_numeric(runs[metric], errors="coerce").dropna().tolist()
            mean, lo, hi = mean_ci(values) if values else (np.nan, np.nan, np.nan)
            rows.append(
                {
                    "approach": approach,
                    "metric": metric,
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values (same order as given). NaN stays NaN and is not counted."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m, running = len(valid), 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def compare_approaches(
    metrics: pd.DataFrame, names: list[str], confidence: float = 0.95
) -> pd.DataFrame:
    """Paired t-test for every pair of approaches on each metric in ``names``.

    ``metrics`` has one row per run with ``approach`` and ``seed`` columns; runs of two
    approaches with the same seed form a pair, and pairs with a missing value are dropped.
    ``difference`` is A minus B, with its confidence interval. P-values are Holm-adjusted
    across the pairs for each metric. With fewer than two pairs, or when every pair differs
    by the same amount (no spread), no test is possible and ``p`` is NaN.
    """
    approaches = list(dict.fromkeys(metrics["approach"]))
    rows = []
    for metric in names:
        if metric not in metrics.columns:
            continue
        values = metrics.assign(value=pd.to_numeric(metrics[metric], errors="coerce"))
        start = len(rows)
        for a, b in combinations(approaches, 2):
            pairs = (
                values[values["approach"] == a][["seed", "value"]]
                .merge(values[values["approach"] == b][["seed", "value"]], on="seed")
                .dropna()
            )
            diff = (pairs["value_x"] - pairs["value_y"]).to_numpy()
            mean, lo, hi = mean_ci(diff.tolist(), confidence) if len(diff) else (np.nan,) * 3
            t = p = np.nan
            if len(diff) >= 2 and np.std(diff) > 0:
                result = stats.ttest_rel(pairs["value_x"], pairs["value_y"])
                t, p = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "metric": metric,
                    "approach_a": a,
                    "approach_b": b,
                    "pairs": len(diff),
                    "difference": mean,
                    "lo": lo,
                    "hi": hi,
                    "t": t,
                    "p": p,
                }
            )
        adjusted = holm([r["p"] for r in rows[start:]])
        for r, p_holm in zip(rows[start:], adjusted, strict=True):
            r["p_holm"] = p_holm
            r["significant"] = bool(p_holm < ALPHA) if not np.isnan(p_holm) else None
    return pd.DataFrame(
        rows,
        columns=[
            "metric", "approach_a", "approach_b", "pairs", "difference", "lo", "hi",
            "t", "p", "p_holm", "significant",
        ],
    )


def write_report(
    tables: dict[str, pd.DataFrame], folder: str | Path, title: str, subfolder: str = "report"
) -> Path:
    """Write each table as CSV and all of them as one HTML page under ``folder/subfolder/``."""
    out = Path(folder) / subfolder
    out.mkdir(parents=True, exist_ok=True)
    parts = [
        f"<h1>{title}</h1>",
        (
            "<p>Synthetic data. Mean ± 95% CI across seeds. Differences: paired t-tests "
            "across seeds, Holm-adjusted per metric.</p>"
        ),
    ]
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        parts.append(f"<h2>{name.replace('_', ' ').capitalize()}</h2>")
        parts.append(table.to_html(index=False, border=0))
    html = out / "report.html"
    html.write_text(
        "<!doctype html><meta charset='utf-8'><title>" + title + "</title>"
        "<style>body{font-family:sans-serif;margin:2rem}td,th{padding:4px 12px;"
        "text-align:left}</style>" + "\n".join(parts)
    )
    return html
