"""Gaussian copula generator (statistical approach), written with NumPy and SciPy.

Every learnable column (see ``fields.py``) gets a marginal distribution. Each row is drawn
as correlated standard normals, which are mapped through each marginal's inverse CDF, so
the columns keep their own distributions while moving together. Nullable columns get a
second dimension for "is this value present", so columns that are empty together (for
example hearing date and hearing type) stay empty together.

- Scenario 1: marginals from the calibration statistics. Correlations come from the little
  the calibration store gives: pairwise rates (for example the defended rate by claim type)
  are matched by choosing the correlation that reproduces them, and columns with identical
  null rates are treated as missing together. Everything else is independent.
- Scenario 2: marginals are the sample's empirical distributions (interpolated, so values
  fall between sample values rather than copying them) and the correlation matrix is
  estimated from the sample's normal scores, shrunk towards independence because 200 rows
  give a noisy estimate.

Categories are placed on the normal scale in a fixed order (most frequent first, or by
the pairwise rate when there is one); the copula can only express monotone relationships
in that order. That limit is part of what the comparison measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from synthgen.generators.fields import Field, assemble, build_fields

# Weight of the identity matrix when shrinking the Scenario 2 correlation estimate
SHRINKAGE = 0.1
# Correlation between "present" dimensions of columns with the same null rate (Scenario 1)
SHARED_MISSING_RHO = 0.99
_EPS = 1e-6


class GaussianCopulaGenerator:
    name = "gaussian_copula"
    supports_scenarios = frozenset({1, 2})

    def __init__(self) -> None:
        self.profile = None
        self.calibration = None
        self.fields: list[Field] = []
        self.dims: list[tuple[Field, str]] = []  # (field, "value" | "present")
        self.corr: np.ndarray | None = None

    def fit(self, profile, calibration, sample: pd.DataFrame | None = None) -> None:
        self.profile, self.calibration = profile, calibration
        self.fields = build_fields(profile, calibration, sample)
        self.dims = []
        for f in self.fields:
            self.dims.append((f, "value"))
            if 0 < f.missing < 1:
                self.dims.append((f, "present"))
        if any(f.learned_from_sample for f in self.fields):
            self.corr = self._fit_correlation()
        else:
            self.corr = self._prior_correlation()

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.profile is None:
            raise RuntimeError("Call fit() before sample()")
        rng = np.random.default_rng(seed)
        d = len(self.dims)
        z = rng.standard_normal((n, d)) @ _matrix_root(self.corr).T if d else np.zeros((n, 0))
        u = np.clip(norm.cdf(z), _EPS, 1 - _EPS)

        draws, present = {}, {}
        for j, (f, role) in enumerate(self.dims):
            if role == "present":
                present[f.name] = u[:, j] >= f.missing
            else:
                draws[f.name] = inverse_cdf(f, u[:, j], z[:, j])
        return assemble(self.profile, self.calibration, self.fields, draws, present, n, seed)

    # -- Scenario 2: estimate from the sample ---------------------------------------------

    def _fit_correlation(self) -> np.ndarray:
        rng = np.random.default_rng(0)  # tie-breaking only; fixed so fits are reproducible
        n = next(len(f.data) for f in self.fields if f.learned_from_sample)
        scores = np.zeros((n, len(self.dims)))
        learned = np.zeros(len(self.dims), dtype=bool)
        for j, (f, role) in enumerate(self.dims):
            if not f.learned_from_sample:
                scores[:, j] = rng.standard_normal(n)
                continue
            learned[j] = True
            if role == "present":
                u = _category_scores(
                    np.where(f.observed_missing, "missing", "present"),
                    ["missing", "present"],
                    np.array([f.missing, 1 - f.missing]),
                    rng,
                )
            elif f.kind == "category":
                u = _category_scores(f.imputed(rng), f.categories, f.probs, rng)
            else:
                u = (rankdata(f.imputed(rng)) - 0.5) / n
            scores[:, j] = norm.ppf(np.clip(u, _EPS, 1 - _EPS))

        corr = np.nan_to_num(np.corrcoef(scores, rowvar=False), nan=0.0)
        corr = np.atleast_2d(corr)
        # Dimensions with no sample data stay independent of everything else
        corr[~learned, :] = 0.0
        corr[:, ~learned] = 0.0
        corr = (1 - SHRINKAGE) * corr + SHRINKAGE * np.eye(len(corr))
        np.fill_diagonal(corr, 1.0)
        return _nearest_correlation(corr)

    # -- Scenario 1: from calibration statistics ------------------------------------------

    def _prior_correlation(self) -> np.ndarray:
        d = len(self.dims)
        corr = np.eye(d)
        index = {(f.name, role): j for j, (f, role) in enumerate(self.dims)}
        by_name = {f.name: f for f in self.fields}

        for (cond, flag), rates in self.calibration.pairwise.items():
            a, b = by_name.get(cond), by_name.get(flag)
            if a is None or b is None or a.kind != "category" or set(b.categories) != {"True", "False"}:
                continue
            _order(a, sorted(a.categories, key=lambda c: rates.get(c, 0.0)))
            _order(b, ["False", "True"])
            rho = _match_pairwise(a, b, rates)
            i, j = index[(a.name, "value")], index[(b.name, "value")]
            corr[i, j] = corr[j, i] = rho

        shared: dict[float, list[int]] = {}
        for f in self.fields:
            if 0 < f.missing < 1:
                shared.setdefault(round(f.missing, 4), []).append(index[(f.name, "present")])
        for group in shared.values():
            for i in group:
                for j in group:
                    if i != j:
                        corr[i, j] = SHARED_MISSING_RHO
        return _nearest_correlation(corr)


# -- helpers --------------------------------------------------------------------------------


def _order(f: Field, categories: list[str]) -> None:
    lookup = dict(zip(f.categories, f.probs, strict=True))
    f.categories = list(categories)
    f.probs = np.array([lookup[c] for c in categories], dtype=float)


def _category_scores(values, categories, probs, rng) -> np.ndarray:
    """Uniform scores: each category owns a slice of [0, 1]; rows spread within their slice."""
    upper = np.cumsum(probs) / np.sum(probs)
    lower = np.concatenate([[0.0], upper[:-1]])
    pos = {c: i for i, c in enumerate(categories)}
    idx = np.array([pos[v] for v in values])
    return lower[idx] + rng.random(len(idx)) * (upper[idx] - lower[idx])


def inverse_cdf(f: Field, u: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Map uniform scores through the field's inverse CDF."""
    if f.kind == "category":
        cum = np.cumsum(f.probs) / np.sum(f.probs)
        idx = np.minimum(np.searchsorted(cum, u, side="right"), len(cum) - 1)
        return np.array(f.categories, dtype=object)[idx]
    if f.learned_from_sample:  # empirical quantiles, interpolated between sample values
        return np.quantile(f.present_values(), u)
    d = f.dist
    if f.kind == "date":
        return np.floor(d["start"] + u * (d["end"] - d["start"] + 1))
    if f.kind == "gap":
        return np.maximum(d["mean"] + d.get("std", 0.0) * z, d.get("min", 1))
    if d.get("dist") == "lognormal":
        return np.exp(d["mean"] + d["std"] * z)
    if "mean" in d:
        return d["mean"] + d["std"] * z
    return f.lo + u * (f.hi - f.lo)


def _match_pairwise(cond: Field, flag: Field, rates: dict[str, float]) -> float:
    """The latent correlation that best reproduces P(flag = True | category).

    Monte Carlo over a grid of correlations, with fixed draws so the result is reproducible.
    """
    rng = np.random.default_rng(0)
    z1, e = rng.standard_normal(100_000), rng.standard_normal(100_000)
    cum = np.cumsum(cond.probs) / np.sum(cond.probs)
    bins = np.minimum(np.searchsorted(cum, norm.cdf(z1), side="right"), len(cum) - 1)
    threshold = norm.ppf(flag.probs[0] / flag.probs.sum())  # above it: True
    targets = np.array([rates.get(c, np.nan) for c in cond.categories])
    weights = np.where(np.isnan(targets), 0.0, cond.probs)
    best, best_loss = 0.0, np.inf
    for rho in np.linspace(-0.95, 0.95, 191):
        z2 = rho * z1 + np.sqrt(1 - rho**2) * e
        hits = np.bincount(bins, weights=z2 > threshold, minlength=len(cum))
        counts = np.maximum(np.bincount(bins, minlength=len(cum)), 1)
        loss = np.nansum(weights * (hits / counts - targets) ** 2)
        if loss < best_loss:
            best, best_loss = float(rho), loss
    return best


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues and rescale to a unit diagonal."""
    values, vectors = np.linalg.eigh((corr + corr.T) / 2)
    fixed = vectors @ np.diag(np.maximum(values, 1e-6)) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    return fixed / np.outer(scale, scale)


def _matrix_root(corr: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(corr)
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
