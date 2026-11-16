import json

import numpy as np
import pandas as pd
import pytest

from synthgen.generators import CalibrationStats, Generator, KnowledgeBasedGenerator
from synthgen.generators.copula import GaussianCopulaGenerator
from synthgen.generators.deep import DeepGenerator
from synthgen.generators.fields import build_fields
from synthgen.generators.llm import LLMGenerator, _allocate
from synthgen.orchestration.orchestrator import run_generator
from synthgen.rules import validate


def _calibration():
    return CalibrationStats(
        categorical={
            "claim_type": {"Money claim": 0.8, "Personal injury": 0.2},
            "status": {"Issued": 0.5, "Defended": 0.3, "Listed": 0.2},
        },
        numeric={"claim_value": {"mean": 7.0, "std": 0.8, "min": 0.01, "max": 100000, "dist": "lognormal"}},
        date_ranges={"issue_date": {"start": "2024-01-01", "end": "2025-06-30"}},
        gaps_days={("issue_date", "hearing_date"): {"mean": 200, "std": 60, "min": 14}},
        null_rate={"hearing_date": 0.3},
    )


def test_all_generators_match_interface():
    for gen in [
        KnowledgeBasedGenerator(),
        GaussianCopulaGenerator(),
        DeepGenerator(),
        LLMGenerator(None, "llama3.1:8b"),
    ]:
        assert isinstance(gen, Generator)


def test_deep_generator_is_scenario_2_only():
    assert DeepGenerator().supports_scenarios == {2}
    assert DeepGenerator("tvae").supports_scenarios == {2}
    with pytest.raises(ValueError):
        DeepGenerator("gan")


def test_knowledge_based_shape_and_reproducible(profile):
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    a, b = gen.sample(200, seed=7), gen.sample(200, seed=7)
    assert list(a.columns) == [c.name for c in profile.columns]
    assert len(a) == 200
    pd.testing.assert_frame_equal(a, b)


def test_orchestrator_returns_only_valid_rows(profile):
    data, record = run_generator(
        KnowledgeBasedGenerator(), profile, _calibration(), scenario=1, n_rows=500, seed=1
    )
    assert len(data) == 500
    assert validate(data, profile).passed.all()
    assert record.complete
    # Both rules are read as generation steps, so nothing needs rejecting
    assert record.rows_rejected == 0


# -- shared fields, Gaussian copula, CTGAN and LLM -----------------------------------------


def _sample(profile, n=200):
    """A fictional Scenario 2 sample: valid knowledge-based rows, as strings like a CSV."""
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(n * 2, seed=99)
    data = data[validate(data, profile).passed].head(n).reset_index(drop=True)
    return data.astype(object).where(data.notna(), None).astype(str).replace("None", None)


def test_fields_skip_identifiers_and_model_dependent_dates(profile):
    fields = {f.name: f for f in build_fields(profile, _calibration())}
    assert "case_number" not in fields  # identifiers are left to the shared fallback
    assert fields["hearing_date"].kind == "gap"
    assert fields["hearing_date"].parent == "issue_date"
    # Calibration categories are kept only where the profile allows them
    assert set(fields["claim_type"].categories) <= {
        "Money claim", "Possession claim", "Personal injury", "Housing disrepair",
    }


@pytest.mark.parametrize("scenario", [1, 2])
def test_copula_shape_reproducible_and_valid(profile, scenario):
    sample = _sample(profile) if scenario == 2 else None
    gen = GaussianCopulaGenerator()
    gen.fit(profile, _calibration(), sample)
    a, b = gen.sample(300, seed=3), gen.sample(300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == [c.name for c in profile.columns]
    data, record = run_generator(
        GaussianCopulaGenerator(), profile, _calibration(), scenario, 300, 1, sample
    )
    assert len(data) == 300 and record.complete
    assert validate(data, profile).passed.all()


def test_copula_matches_calibration_marginals(profile):
    gen = GaussianCopulaGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(5000, seed=1)
    share = (data["claim_type"] == "Money claim").mean()
    assert abs(share - 0.8) < 0.03
    assert abs(data["hearing_date"].isna().mean() - 0.3) < 0.03
    hearing = data.dropna(subset=["hearing_date"])
    assert (hearing["hearing_date"] > hearing["issue_date"]).all()


def test_copula_reproduces_a_pairwise_rate(profile):
    """Scenario 1: the copula picks the correlation that matches a calibration pairwise rate."""
    profile.columns[6].allowed_values = None  # status: use the calibration's categories
    cal = _calibration()
    cal.categorical["status"] = {"True": 0.3, "False": 0.7}  # treat status as a yes/no flag
    cal.pairwise = {("claim_type", "status"): {"Money claim": 0.2, "Personal injury": 0.7}}
    gen = GaussianCopulaGenerator()
    gen.fit(profile, cal)
    data = gen.sample(20_000, seed=1)
    rates = (data["status"] == "True").groupby(data["claim_type"]).mean()
    assert rates["Personal injury"] - rates["Money claim"] > 0.3


def test_copula_learns_a_sample_correlation(profile):
    sample = _sample(profile, n=400)
    # Make claim value depend strongly on claim type in the sample
    bump = sample["claim_type"].eq("Personal injury").map({True: 50_000.0, False: 0.0})
    sample["claim_value"] = (pd.to_numeric(sample["claim_value"]) * 0.01 + bump).round(2).astype(str)
    gen = GaussianCopulaGenerator()
    gen.fit(profile, _calibration(), sample)
    data = gen.sample(4000, seed=2)
    high = (pd.to_numeric(data["claim_value"]) > 1000).groupby(data["claim_type"]).mean()
    # A copula keeps the direction but blurs a step change (independence gives ~0.2 for both)
    assert high["Personal injury"] > high["Money claim"] + 0.15


def test_ctgan_needs_sample_rows(profile):
    with pytest.raises(ValueError):
        DeepGenerator().fit(profile, _calibration(), None)


def test_ctgan_trains_samples_and_is_reproducible(profile):
    pytest.importorskip("torch")
    sample = _sample(profile)
    outputs = []
    for _ in range(2):
        gen = DeepGenerator("ctgan", epochs=5, seed=4)
        gen.fit(profile, _calibration(), sample)
        outputs.append(gen.sample(200, seed=8))
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    assert list(outputs[0].columns) == [c.name for c in profile.columns]
    assert len(gen.loss_history) == 5
    allowed = set(profile.column("claim_type").allowed_values)
    assert set(outputs[0]["claim_type"].dropna()) <= allowed


def test_tvae_trains_samples_and_is_reproducible(profile):
    pytest.importorskip("torch")
    sample = _sample(profile)
    outputs = []
    for _ in range(2):
        gen = DeepGenerator("tvae", epochs=20, seed=4)
        gen.fit(profile, _calibration(), sample)
        outputs.append(gen.sample(300, seed=8))
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    assert list(outputs[0].columns) == [c.name for c in profile.columns]
    assert len(gen.loss_history) == 20
    assert set(gen.loss_history[0]) == {"epoch", "reconstruction_loss", "kl_loss"}
    allowed = set(profile.column("claim_type").allowed_values)
    assert set(outputs[0]["claim_type"].dropna()) <= allowed
    # Categories are drawn, not just the most likely one: both claim types appear
    assert outputs[0]["claim_type"].nunique() > 1


def test_tvae_runs_through_the_orchestrator(profile):
    pytest.importorskip("torch")
    data, record = run_generator(
        DeepGenerator("tvae", epochs=20, seed=1), profile, _calibration(), 2, 200, 1,
        _sample(profile),
    )
    assert len(data) == 200 and record.complete
    assert validate(data, profile).passed.all()


class _FakeGateway:
    """Answers with schema-valid rows and records each prompt (no model server needed)."""

    def __init__(self, bad_first: bool = False):
        self.prompts, self.bad_first = [], bad_first

    def complete_json(self, request):
        self.prompts.append(request.prompt)
        if self.bad_first and len(self.prompts) == 1:
            return {"rows": "not a list"}
        size = request.json_schema["properties"]["rows"]["maxItems"]
        rows = [
            {
                "claim_type": "Money claim",
                "claim_value": 1234.567,
                "issue_date": "2024-03-01",
                "hearing_date": "2024-09-01" if i % 2 else None,
                "court": "Leeds",
                "status": "Listed" if i % 2 else "Issued",
            }
            for i in range(size)
        ]
        return {"rows": rows}


def test_llm_generator_batches_counts_and_ids(profile):
    gateway = _FakeGateway()
    gen = LLMGenerator(gateway, "llama3.1:8b", batch_size=20)
    gen.fit(profile, _calibration())
    data = gen.sample(50, seed=1)
    assert len(data) == 50 and len(gateway.prompts) == 3  # 20 + 20 + 10
    assert list(data.columns) == [c.name for c in profile.columns]
    assert data["case_number"].is_unique  # identifiers are filled in code
    assert "case_number" not in gateway.prompts[0]
    assert data["claim_value"].iloc[0] == 1234.57  # coerced to the profile's decimals
    # The first batch asks for the target mix; later batches correct the drift
    assert '"Money claim" ×16' in gateway.prompts[0]
    # Batch 1 came back all "Money claim" (fake model), so batch 2 makes up the shortfall:
    # 40 rows at 80/20 means 32 and 8 in total, of which 20 and 0 already exist
    assert '"Money claim" ×12, "Personal injury" ×8' in gateway.prompts[1]
    assert "Hearing date must be after issue date" in gateway.prompts[0]


def test_llm_generator_reports_progress_before_each_call(profile):
    gateway = _FakeGateway()
    gen = LLMGenerator(gateway, "llama3.1:8b", batch_size=20)
    gen.fit(profile, _calibration())
    seen = []
    gen.on_batch = lambda have, want: seen.append((have, want, len(gateway.prompts)))
    gen.sample(50, seed=1)
    # Called before each of the three calls, so a stop raised here skips the next call
    assert seen == [(0, 50, 0), (20, 50, 1), (40, 50, 2)]


def test_llm_generator_stops_between_batches_when_asked(profile):
    class Stop(Exception):
        pass

    def stop_after_first(have, want):
        if have:
            raise Stop

    gateway = _FakeGateway()
    gen = LLMGenerator(gateway, "llama3.1:8b", batch_size=20)
    gen.fit(profile, _calibration())
    gen.on_batch = stop_after_first
    with pytest.raises(Stop):
        gen.sample(50, seed=1)
    assert len(gateway.prompts) == 1  # no further model calls after the stop


def test_llm_generator_retries_unusable_replies(profile):
    gateway = _FakeGateway(bad_first=True)
    gen = LLMGenerator(gateway, "llama3.1:8b", batch_size=10)
    gen.fit(profile, _calibration())
    assert len(gen.sample(10, seed=1)) == 10
    assert len(gateway.prompts) == 2


def test_llm_prompt_includes_sample_examples_in_scenario_2(profile):
    gateway = _FakeGateway()
    gen = LLMGenerator(gateway, "llama3.1:8b", batch_size=10)
    gen.fit(profile, _calibration(), _sample(profile))
    gen.sample(10, seed=1)
    examples = gateway.prompts[0].split("do not copy):")[1].strip().splitlines()[:5]
    assert all("case_number" not in json.loads(line) for line in examples)


def test_allocate_is_exact():
    assert _allocate([0.5, 0.3, 0.2], 7) == [4, 2, 1]
    assert sum(_allocate([0.0, 0.0], 5)) == 5
    assert np.array_equal(_allocate([1, 1], 0), [0, 0])
