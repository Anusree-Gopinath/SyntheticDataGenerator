"""TVAE (Xu et al., 2019, "Modeling Tabular Data using Conditional GAN") in PyTorch.

The variational autoencoder from the same paper as CTGAN, sharing its representation
(``encode_table`` in ``ctgan.py``): continuous columns by mode-specific normalisation,
discrete columns one-hot.

- An encoder maps a row to a Gaussian over a 128-number latent code; a decoder maps a
  latent code back to a row.
- Loss: reconstruction plus the KL divergence from a standard normal. A position within a
  mode is scored as a Gaussian whose spread (one per position) is learned; a mode or a
  category is scored by cross-entropy. Reconstruction is weighted by 2, as published.
- Sampling: latent codes from a standard normal are decoded; positions get noise with the
  learned spread, and modes and categories are drawn from the decoder's probabilities
  (taking the most likely value all but removes rare categories on a small sample).

Written here for the same reasons as CTGAN (see docs/decisions.md).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from synthgen.generators.ctgan import decode_table, encode_table

EMBEDDING = 128
LAYERS = (128, 128)
LOSS_FACTOR = 2.0
SIGMA_RANGE = (0.01, 1.0)


class _Encoder(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        layers, width = [], inputs
        for size in LAYERS:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        self.net = nn.Sequential(*layers)
        self.mu, self.logvar = nn.Linear(width, EMBEDDING), nn.Linear(width, EMBEDDING)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    def __init__(self, outputs: int):
        super().__init__()
        layers, width = [], EMBEDDING
        for size in reversed(LAYERS):
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, outputs))
        self.net = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.full((outputs,), 0.1))

    def forward(self, z):
        return self.net(z)


class TVAE:
    """Fit on a table of discrete (category codes) and continuous columns; sample new rows.

    Same inputs and outputs as ``CTGAN``.
    """

    def __init__(
        self, epochs: int = 300, batch_size: int = 500, seed: int = 0, draw_categories: bool = True
    ):
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.draw_categories = draw_categories
        self.loss_history: list[dict[str, float]] = []

    def fit(self, discrete: dict[str, tuple[np.ndarray, int]], continuous: dict[str, np.ndarray]):
        if not discrete and not continuous:
            raise ValueError("TVAE needs at least one column to learn")
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        table = encode_table(discrete, continuous, self.seed, rng)
        self.spans, self.layout = table.spans, table.layout
        data = torch.from_numpy(table.data)
        n, width = data.shape

        self.encoder, self.decoder = _Encoder(width), _Decoder(width)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimiser = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
        batch = max(1, min(self.batch_size, n))
        steps = max(1, n // batch)

        self.loss_history = []
        for epoch in range(self.epochs):
            order = torch.from_numpy(rng.permutation(n))
            rec_total = kl_total = 0.0
            for step in range(steps):
                real = data[order[step * batch : (step + 1) * batch]]
                mu, logvar = self.encoder(real)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                rec = self._reconstruction(self.decoder(z), real)
                kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / real.size(0)
                optimiser.zero_grad()
                (rec + kl).backward()
                optimiser.step()
                self.decoder.sigma.data.clamp_(*SIGMA_RANGE)
                rec_total += rec.item()
                kl_total += kl.item()
            self.loss_history.append(
                {
                    "epoch": epoch + 1,
                    "reconstruction_loss": round(rec_total / steps, 4),
                    "kl_loss": round(kl_total / steps, 4),
                }
            )
        return self

    def sample(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Category codes for discrete columns and values for continuous ones."""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.decoder.eval()
        with torch.no_grad():
            raw = self.decoder(torch.randn(n, EMBEDDING)).numpy()
            sigma = self.decoder.sigma.numpy()
        self.decoder.train()
        for span in self.spans:
            i, w = span.start, span.width
            if span.kind == "tanh":
                raw[:, i] = np.clip(np.tanh(raw[:, i]) + rng.normal(0, sigma[i], n), -1, 1)
            elif self.draw_categories and w > 1:
                # Draw from the decoder's probabilities rather than taking the most likely
                # value, which all but removes rare categories on a small sample
                logits = raw[:, i : i + w] - raw[:, i : i + w].max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                picks = (probs.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1).clip(max=w - 1)
                raw[:, i : i + w] = np.eye(w, dtype=raw.dtype)[picks]
        return decode_table(raw, self.layout)

    def _reconstruction(self, recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(())
        sigma = self.decoder.sigma
        for span in self.spans:
            i, w = span.start, span.width
            if span.kind == "tanh":
                std = sigma[i]
                loss = loss + ((real[:, i] - torch.tanh(recon[:, i])) ** 2 / 2 / std**2).sum()
                loss = loss + torch.log(std) * real.size(0)
            else:
                target = real[:, i : i + w].argmax(dim=1)
                loss = loss + F.cross_entropy(recon[:, i : i + w], target, reduction="sum")
        return loss * LOSS_FACTOR / real.size(0)
