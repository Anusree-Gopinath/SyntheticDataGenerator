"""The common generator interface.

Every approach implements ``fit`` and ``sample`` so the orchestrator, validator and
evaluators treat them identically. Generators receive the calibration store only
(and the fictional sample in Scenario 2); they never see the evaluation store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from synthgen.core.models import SchemaProfile


@dataclass
class CalibrationStats:
    """Aggregate statistics a generator is allowed to see.

    - ``categorical``: column -> {value: proportion}
    - ``numeric``: column -> {"mean", "std", "min", "max"} (optionally "dist": "lognormal")
    - ``date_ranges``: column -> {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    - ``gaps_days``: (from_col, to_col) -> {"mean", "std", "min"} for dependent dates
    - ``null_rate``: column -> share of empty values
    - ``pairwise``: (category column, yes/no column) -> {category: share of "True"}
    - ``conditional``: (parent column, child column) -> {parent value: {child value: share}}
    """

    categorical: dict[str, dict[str, float]] = field(default_factory=dict)
    numeric: dict[str, dict[str, float]] = field(default_factory=dict)
    date_ranges: dict[str, dict[str, str]] = field(default_factory=dict)
    gaps_days: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    null_rate: dict[str, float] = field(default_factory=dict)
    pairwise: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    conditional: dict[tuple[str, str], dict[str, dict[str, float]]] = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    name: str
    supports_scenarios: frozenset[int]

    def fit(
        self,
        profile: SchemaProfile,
        calibration: CalibrationStats,
        sample: pd.DataFrame | None = None,
    ) -> None: ...

    def sample(self, n: int, seed: int) -> pd.DataFrame: ...
