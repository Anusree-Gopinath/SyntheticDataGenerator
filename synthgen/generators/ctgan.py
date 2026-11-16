"""CTGAN (Xu et al., 2019, "Modeling Tabular Data using Conditional GAN") in PyTorch.

A compact implementation of the published method, trained on the Scenario 2 sample:

- Mode-specific normalisation: each continuous column is modelled with a variational
  Gaussian mixture; a value is encoded as its mode (one-hot) plus its position within the
  mode (a scalar in [-1, 1]).
- Discrete columns are one-hot. A conditional vector picks one discrete column and one of
  its categories per row, and training-by-sampling (categories drawn by log-frequency)
  makes the model learn rare categories too.
- A residual generator and a PacGAN discriminator, trained as a WGAN with gradient penalty.

Written here rather than taken from a library so it runs with this project's pinned pandas
and NumPy versions and adds no dependency beyond PyTorch and scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture
from torch import nn
from torch.nn import functional as F

MAX_MODES = 10
MODE_WEIGHT_MIN = 0.005
EMBEDDING = 128
LAYERS = (256, 256)
PAC = 10
GP_WEIGHT = 10.0
TAU = 0.2


@dataclass
class _Span:
    start: int
    width: int
    kind: str  # "tanh" or "softmax"


class _Continuous:
    """Mode-specific normalisation for one column."""

    def __init__(self, values: np.ndarray, seed: int):
        self.lo, self.hi = float(values.min()), float(values.max())
        distinct = len(np.unique(values))
        if distinct < 2:
            self.means, self.stds = np.array([self.lo]), np.array([1.0])
            self.mixture = None
            return
        mixture = BayesianGaussianMixture(
            n_components=min(MAX_MODES, distinct),
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=200,
            random_state=seed,
        ).fit(values.reshape(-1, 1))
        self.keep = mixture.weights_ > MODE_WEIGHT_MIN
        self.means = mixture.means_.ravel()[self.keep]
        self.stds = np.sqrt(mixture.covariances_.ravel()[self.keep])
        self.stds = np.maximum(self.stds, 1e-3 * max(self.hi - self.lo, 1.0))
        self.mixture = mixture

    @property
    def width(self) -> int:
        return 1 + len(self.means)

    def encode(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(values)
        if self.mixture is None:
            modes = np.zeros(n, dtype=int)
        else:
            probs = self.mixture.predict_proba(values.reshape(-1, 1))[:, self.keep] + 1e-6
            probs /= probs.sum(axis=1, keepdims=True)
            cum = probs.cumsum(axis=1)
            modes = np.minimum((cum < rng.random((n, 1))).sum(axis=1), len(self.means) - 1)
        alpha = np.clip((values - self.means[modes]) / (4 * self.stds[modes]), -0.99, 0.99)
        out = np.zeros((n, self.width), dtype=np.float32)
        out[:, 0] = alpha
        out[np.arange(n), 1 + modes] = 1.0
        return out

    def decode(self, block: np.ndarray) -> np.ndarray:
        modes = block[:, 1:].argmax(axis=1)
        values = block[:, 0] * 4 * self.stds[modes] + self.means[modes]
        return np.clip(values, self.lo, self.hi)


@dataclass
class EncodedTable:
    data: np.ndarray  # one row per sample row, columns laid out as in ``layout``
    spans: list[_Span]  # how each part of a row is activated
    layout: list[tuple]  # ("continuous", name, start, encoder) or ("discrete", name, start, k)
    discrete_columns: list[tuple]  # (name, start in data, categories, start in condition)


def encode_table(
    discrete: dict[str, tuple[np.ndarray, int]],
    continuous: dict[str, np.ndarray],
    seed: int,
    rng: np.random.Generator,
) -> EncodedTable:
    """Continuous columns by mode-specific normalisation, then discrete columns one-hot.
    Shared by CTGAN and TVAE, which use the same representation."""
    blocks, spans, layout, discrete_columns = [], [], [], []
    width = cond_width = 0
    for name, values in continuous.items():
        enc = _Continuous(np.asarray(values, dtype=float), seed)
        blocks.append(enc.encode(np.asarray(values, dtype=float), rng))
        spans += [_Span(width, 1, "tanh"), _Span(width + 1, enc.width - 1, "softmax")]
        layout.append(("continuous", name, width, enc))
        width += enc.width
    for name, (codes, k) in discrete.items():
        block = np.zeros((len(codes), k), dtype=np.float32)
        block[np.arange(len(codes)), codes] = 1.0
        blocks.append(block)
        spans.append(_Span(width, k, "softmax"))
        layout.append(("discrete", name, width, k))
        discrete_columns.append((name, width, k, cond_width))
        width += k
        cond_width += k
    data = np.concatenate(blocks, axis=1).astype(np.float32)
    return EncodedTable(data, spans, layout, discrete_columns)


def decode_table(raw: np.ndarray, layout: list[tuple]) -> dict[str, np.ndarray]:
    """Rows in the encoded layout (positions already in [-1, 1]) back to codes and values."""
    out: dict[str, np.ndarray] = {}
    for kind, name, start, extra in layout:
        if kind == "discrete":
            out[name] = raw[:, start : start + extra].argmax(axis=1)
        else:
            out[name] = extra.decode(raw[:, start : start + extra.width])
    return out


class _Residual(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.fc, self.bn = nn.Linear(i, o), nn.BatchNorm1d(o)

    def forward(self, x):
        return torch.cat([F.relu(self.bn(self.fc(x))), x], dim=1)


class _Generator(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        layers, width = [], inputs
        for size in LAYERS:
            layers.append(_Residual(width, size))
            width += size
        layers.append(nn.Linear(width, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Discriminator(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.inputs = inputs
        layers, width = [], inputs * PAC
        for size in LAYERS:
            layers += [nn.Linear(width, size), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(-1, self.inputs * PAC))

    def gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0) // PAC, 1, 1).repeat(1, PAC, real.size(1))
        alpha = alpha.view(-1, real.size(1))
        mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        grads = torch.autograd.grad(
            self(mixed).sum(), mixed, create_graph=True, retain_graph=True
        )[0]
        norms = grads.view(-1, PAC * real.size(1)).norm(2, dim=1)
        return GP_WEIGHT * ((norms - 1) ** 2).mean()


class CTGAN:
    """Fit on a table of discrete (category codes) and continuous columns; sample new rows.

    ``discrete``: name -> (integer codes, number of categories).
    ``continuous``: name -> float values (no gaps).
    """

    def __init__(self, epochs: int = 300, batch_size: int = 500, seed: int = 0):
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.loss_history: list[dict[str, float]] = []

    def fit(self, discrete: dict[str, tuple[np.ndarray, int]], continuous: dict[str, np.ndarray]):
        if not discrete and not continuous:
            raise ValueError("CTGAN needs at least one column to learn")
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        table = encode_table(discrete, continuous, self.seed, rng)
        data, self.spans, self.layout = table.data, table.spans, table.layout
        self.discrete_columns = table.discrete_columns
        width = self.data_width = data.shape[1]
        cond_width = self.cond_width = sum(k for _, _, k, _ in self.discrete_columns)
        n = self.n_rows = len(data)

        # Training-by-sampling: rows per (column, category); categories drawn by log-frequency
        self.rows_by_category = [
            [np.flatnonzero(data[:, start + c] == 1) for c in range(k)]
            for _, start, k, _ in self.discrete_columns
        ]
        self.log_freq = []
        self.freq = []
        for rows in self.rows_by_category:
            counts = np.array([len(r) for r in rows], dtype=float)
            self.freq.append(counts / counts.sum())
            logs = np.log(counts + 1)
            self.log_freq.append(logs / logs.sum())

        batch = max(PAC, min(self.batch_size, n) // PAC * PAC)
        steps = max(1, n // batch)
        self.generator = _Generator(EMBEDDING + cond_width, width)
        disc = _Discriminator(width + cond_width)
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )
        opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
        tensor = torch.from_numpy(data)

        self.loss_history = []
        for epoch in range(self.epochs):
            g_total = d_total = 0.0
            for _ in range(steps):
                # Discriminator step
                cond, cols, cats = self._condition(batch, rng, self.log_freq)
                real = tensor[self._rows(cols, cats, rng, batch)]
                fake = self._activate(self.generator(self._noise(batch, cond)))
                real_in, fake_in = torch.cat([real, cond], 1), torch.cat([fake, cond], 1)
                penalty = disc.gradient_penalty(real_in, fake_in)
                loss_d = disc(fake_in).mean() - disc(real_in).mean()
                opt_d.zero_grad(set_to_none=False)
                penalty.backward(retain_graph=True)
                loss_d.backward()
                opt_d.step()

                # Generator step
                cond, cols, cats = self._condition(batch, rng, self.log_freq)
                raw = self.generator(self._noise(batch, cond))
                fake_in = torch.cat([self._activate(raw), cond], 1)
                loss_g = -disc(fake_in).mean() + self._condition_loss(raw, cols, cats)
                opt_g.zero_grad(set_to_none=False)
                loss_g.backward()
                opt_g.step()
                g_total += loss_g.item()
                d_total += loss_d.item()
            self.loss_history.append(
                {
                    "epoch": epoch + 1,
                    "generator_loss": round(g_total / steps, 4),
                    "discriminator_loss": round(d_total / steps, 4),
                }
            )
        return self

    def sample(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Category codes for discrete columns and values for continuous ones."""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.generator.eval()
        with torch.no_grad():
            cond, _, _ = self._condition(n, rng, self.freq)
            raw = self.generator(self._noise(n, cond)).numpy()
        self.generator.train()
        raw[:, [s.start for s in self.spans if s.kind == "tanh"]] = np.tanh(
            raw[:, [s.start for s in self.spans if s.kind == "tanh"]]
        )
        return decode_table(raw, self.layout)

    # -- helpers ---------------------------------------------------------------------------

    def _noise(self, n: int, cond: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.randn(n, EMBEDDING), cond], dim=1)

    def _condition(self, n: int, rng: np.random.Generator, probs: list[np.ndarray]):
        cond = np.zeros((n, self.cond_width), dtype=np.float32)
        if not self.discrete_columns:
            return torch.from_numpy(cond), None, None
        cols = rng.integers(0, len(self.discrete_columns), size=n)
        cats = np.empty(n, dtype=int)
        for j, (_, _, k, cond_start) in enumerate(self.discrete_columns):
            rows = np.flatnonzero(cols == j)
            cats[rows] = rng.choice(k, size=len(rows), p=probs[j])
            cond[rows, cond_start + cats[rows]] = 1.0
        return torch.from_numpy(cond), cols, cats

    def _rows(self, cols, cats, rng: np.random.Generator, n: int) -> np.ndarray:
        """Real rows matching each condition (any rows when there are no discrete columns)."""
        if cols is None:
            return rng.integers(0, self.n_rows, size=n)
        return np.array(
            [rng.choice(self.rows_by_category[j][c]) for j, c in zip(cols, cats, strict=True)]
        )

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        parts = []
        for span in self.spans:
            x = raw[:, span.start : span.start + span.width]
            parts.append(torch.tanh(x) if span.kind == "tanh" else F.gumbel_softmax(x, tau=TAU))
        return torch.cat(parts, dim=1)

    def _condition_loss(self, raw: torch.Tensor, cols, cats) -> torch.Tensor:
        if cols is None:
            return torch.zeros(())
        loss = torch.zeros(())
        for j, (_, start, k, _) in enumerate(self.discrete_columns):
            rows = np.flatnonzero(cols == j)
            if len(rows):
                logits = raw[rows, start : start + k]
                target = torch.from_numpy(cats[rows])
                loss = loss + F.cross_entropy(logits, target, reduction="sum")
        return loss / raw.size(0)
