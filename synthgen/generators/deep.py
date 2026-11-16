"""Deep generative models (Scenario 2 only): CTGAN and TVAE.

With no rows (Scenario 1) these models could only relearn the prior, so they are
excluded there (design decision D3).

Both learn every column the sample covers (see ``fields.py``): categories and "is this
value present" flags as discrete columns; numbers, dates and gaps between dependent dates
as continuous columns. Columns the sample leaves empty are filled the same way as the
other approaches fill them. Hyperparameters are fixed (the published defaults, with a
smaller batch and more epochs for a 200-row sample; strongly skewed positive numbers are
learned on a log scale) and the training loss is kept for every run
(``loss_history``).

CTGAN is a conditional GAN, TVAE a variational autoencoder; both come from Xu et al. (2019)
and share one representation of the columns (``ctgan.py``), so they differ only in the model.

Requires PyTorch and scikit-learn: ``pip install -e ".[deep]"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew

from synthgen.generators.copula import inverse_cdf
from synthgen.generators.fields import Field, assemble, build_fields

# The published default (300) under-trains on a 200-row sample. CTGAN: amounts came out far
# too high; 1000 matched the sample's quantiles, about 40 s per fit. TVAE: 2000 came
# closest to the sample's quantiles, about 20 s per fit (see docs/decisions.md).
EPOCHS = {"ctgan": 1000, "tvae": 2000}
# Small samples: four steps per epoch rather than one full-sample batch
STEPS_PER_EPOCH = 4
# Positive numbers more skewed than this are learned on a log scale (claim amounts, fees)
LOG_SKEW = 1.0


class DeepGenerator:
    supports_scenarios = frozenset({2})

    def __init__(self, model: str = "ctgan", epochs: int | None = None, seed: int = 0):
        if model not in EPOCHS:
            raise ValueError("model must be 'ctgan' or 'tvae'")
        self.name = model
        self.epochs = epochs if epochs is not None else EPOCHS[model]
        self.seed = seed
        self.loss_history: list[dict[str, float]] = []
        self.logged: set[str] = set()
        self._net = None

    def fit(self, profile, calibration, sample: pd.DataFrame | None = None) -> None:
        if sample is None or sample.empty:
            raise ValueError("Deep generators need sample rows (Scenario 2 only)")

        self.profile, self.calibration = profile, calibration
        self.fields = build_fields(profile, calibration, sample)
        self.learned = [f for f in self.fields if f.learned_from_sample]
        if not self.learned:
            raise ValueError("The sample has no values for any column in this profile")

        self.logged = {
            f.name
            for f in self.learned
            if f.kind == "number" and f.present_values().min() > 0
            and skew(f.present_values()) > LOG_SKEW
        }
        rng = np.random.default_rng(self.seed)
        discrete: dict[str, tuple[np.ndarray, int]] = {}
        continuous: dict[str, np.ndarray] = {}
        for f in self.learned:
            if f.kind == "category":
                pos = {c: i for i, c in enumerate(f.categories)}
                discrete[f.name] = (np.array([pos[v] for v in f.imputed(rng)]), len(f.categories))
            else:
                values = f.imputed(rng)
                continuous[f.name] = np.log(values) if f.name in self.logged else values
            if 0 < f.missing < 1:
                discrete[_present(f)] = (np.asarray(~f.observed_missing, dtype=int), 2)

        n = len(sample)
        batch = max(10, n // STEPS_PER_EPOCH)
        # Imported here so PyTorch loads only when a deep model is used
        if self.name == "ctgan":
            from synthgen.generators.ctgan import CTGAN as Model
        else:
            from synthgen.generators.tvae import TVAE as Model
        self._net = Model(epochs=self.epochs, batch_size=batch, seed=self.seed)
        self._net.fit(discrete, continuous)
        self.loss_history = self._net.loss_history

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self._net is None:
            raise RuntimeError("Call fit() before sample()")
        raw = self._net.sample(n, seed)
        rng = np.random.default_rng(seed)
        draws, present = {}, {}
        for f in self.fields:
            if f.learned_from_sample:
                values = np.exp(raw[f.name]) if f.name in self.logged else raw[f.name]
                draws[f.name] = (
                    np.array(f.categories, dtype=object)[values] if f.kind == "category" else values
                )
                if _present(f) in raw:
                    present[f.name] = raw[_present(f)] == 1
            else:  # not in the sample: independent draws from its calibration marginal
                draws[f.name], present[f.name] = _independent(f, n, rng)
        return assemble(self.profile, self.calibration, self.fields, draws, present, n, seed)


def _present(f: Field) -> str:
    return f"__present__{f.name}"


def _independent(f: Field, n: int, rng: np.random.Generator):
    z = rng.standard_normal(n)
    return inverse_cdf(f, norm.cdf(z), z), rng.random(n) >= f.missing
