"""The generation approaches the review app and the experiment runner can choose from.

``make`` builds a fresh generator for one run; the seed lets approaches that train (CTGAN,
TVAE) start from a different, reproducible state per seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from synthgen.gateway import LLMGateway
from synthgen.generators import Generator, KnowledgeBasedGenerator
from synthgen.generators.copula import GaussianCopulaGenerator
from synthgen.generators.deep import DeepGenerator
from synthgen.generators.llm import LLMGenerator


@dataclass(frozen=True)
class Approach:
    key: str
    label: str
    description: str
    scenarios: frozenset[int]
    # (gateway, model tags from config/models.yaml, seed) -> a fresh generator
    make: Callable[[LLMGateway, dict[str, str], int], Generator]
    model_size: str | None = None  # key in config/models.yaml for local LLM approaches


APPROACHES = [
    Approach(
        "knowledge_based",
        "Knowledge-based",
        "Baseline · rules and calibration statistics",
        frozenset({1, 2}),
        make=lambda gateway, models, seed: KnowledgeBasedGenerator(),
    ),
    Approach(
        "gaussian_copula",
        "Gaussian copula",
        "Statistical · each column's distribution plus correlations between columns",
        frozenset({1, 2}),
        make=lambda gateway, models, seed: GaussianCopulaGenerator(),
    ),
    Approach(
        "ctgan",
        "CTGAN",
        "Deep learning · Scenario 2 only · trains on the sample (about 40 s per seed)",
        frozenset({2}),
        make=lambda gateway, models, seed: DeepGenerator("ctgan", seed=seed),
    ),
    Approach(
        "tvae",
        "TVAE",
        "Deep learning · Scenario 2 only · variational autoencoder on the sample (about 20 s per seed)",
        frozenset({2}),
        make=lambda gateway, models, seed: DeepGenerator("tvae", seed=seed),
    ),
    Approach(
        "llm_small",
        "LLM · small (local)",
        "Batches of up to 50 rows · counts corrected each batch · minutes per batch",
        frozenset({1, 2}),
        make=lambda gateway, models, seed: LLMGenerator(gateway, models["small"]),
        model_size="small",
    ),
]
