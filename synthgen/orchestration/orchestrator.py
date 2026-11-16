"""Run one generator for one scenario and seed, with rejection and regeneration."""

from __future__ import annotations

import time

import pandas as pd

from synthgen.core.models import RunRecord, SchemaProfile
from synthgen.generators.base import CalibrationStats, Generator
from synthgen.rules.validator import validate

MAX_ROUNDS = 5


def run_generator(
    generator: Generator,
    profile: SchemaProfile,
    calibration: CalibrationStats,
    scenario: int,
    n_rows: int,
    seed: int,
    sample: pd.DataFrame | None = None,
    code_version: str | None = None,
) -> tuple[pd.DataFrame, RunRecord]:
    """Generate ``n_rows`` valid rows. Failing rows are dropped and replaced.

    If the target can't be reached within MAX_ROUNDS, the run is marked incomplete
    rather than silently returning fewer rows.
    """
    if scenario not in generator.supports_scenarios:
        raise ValueError(f"{generator.name} does not support scenario {scenario}")
    if scenario == 1:
        sample = None

    start = time.perf_counter()
    generator.fit(profile, calibration, sample)

    kept: list[pd.DataFrame] = []
    rejected = 0
    needed = n_rows
    for round_no in range(MAX_ROUNDS):
        batch = generator.sample(needed, seed=seed + round_no * 10_007)
        result = validate(batch, profile)
        rejected += result.n_failed
        kept.append(batch[result.passed])
        have = sum(len(k) for k in kept)
        if have >= n_rows:
            break
        needed = n_rows - have

    data = pd.concat(kept, ignore_index=True).head(n_rows) if kept else pd.DataFrame()
    # Uniqueness across batches must hold for the final dataset too
    final = validate(data, profile)
    data = data[final.passed].reset_index(drop=True)

    record = RunRecord(
        run_id=f"{profile.profile_id}-{generator.name}-s{scenario}-seed{seed}",
        profile=f"{profile.profile_id}@{profile.version}",
        scenario=scenario,
        approach=generator.name,
        seed=seed,
        rows_requested=n_rows,
        rows_rejected=rejected,
        code_version=code_version,
        duration_seconds=round(time.perf_counter() - start, 2),
        complete=len(data) >= n_rows,
    )
    return data, record
