"""Significance tests in the report and the one-command experiment runner."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from synthgen.evaluation.report import build_report, compare_approaches, holm

ROOT = Path(__file__).resolve().parents[1]


def test_holm_adjusts_in_rank_order_and_keeps_nan():
    adjusted = holm([0.01, 0.04, np.nan, 0.03])
    # three valid p-values: 0.01*3, then max(0.03, 0.03*2), then max(0.06, 0.04*1)
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[3] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert np.isnan(adjusted[2])


def test_compare_approaches_pairs_runs_by_seed():
    runs = pd.DataFrame(
        {
            "approach": ["A"] * 5 + ["B"] * 5,
            "seed": [1, 2, 3, 4, 5] * 2,
            # B is A plus about 1 in every seed, listed in a different order
            "gap": [1.0, 2.0, 3.0, 4.0, 5.0, 6.1, 5.0, 3.9, 3.1, 1.9],
        }
    )
    runs.loc[5:, "seed"] = [5, 4, 3, 2, 1]
    result = compare_approaches(runs, ["gap"]).iloc[0]
    assert (result["approach_a"], result["approach_b"], result["pairs"]) == ("A", "B", 5)
    assert result["difference"] == pytest.approx(-1.0)
    assert result["p"] < 0.001 and bool(result["significant"])


def test_compare_approaches_without_spread_or_pairs_has_no_test():
    runs = pd.DataFrame(
        {"approach": ["A", "A", "B", "B", "C"], "seed": [1, 2, 1, 2, 1],
         "x": [1.0, 2.0, 2.0, 3.0, 9.0]}
    )
    result = compare_approaches(runs, ["x", "missing_metric"])
    result = result.set_index(result["approach_a"] + result["approach_b"])
    assert len(result) == 3  # A-B, A-C, B-C; the missing metric is skipped
    assert np.isnan(result.loc["AB", "p"]) and result.loc["AB", "significant"] is None
    assert result.loc["AB", "difference"] == pytest.approx(-1.0)  # constant difference
    assert result.loc["AC", "pairs"] == 1 and np.isnan(result.loc["AC", "p"])


def test_build_report_ignores_seed_and_run_id():
    runs = pd.DataFrame(
        {"run_id": ["a1", "a2"], "approach": ["A", "A"], "seed": [1, 2], "gap": [1.0, 3.0]}
    )
    assert set(build_report(runs)["metric"]) == {"gap"}


def test_plan_approaches_match_the_app_and_use_the_plan_model():
    from synthgen.orchestration.run_experiments import plan_approaches

    plan = yaml.safe_load((ROOT / "config" / "experiment_plan.yaml").read_text())
    models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    planned = {p.approach.key: p for p in plan_approaches(plan, models)}
    assert set(planned) == {"knowledge_based", "gaussian_copula", "ctgan", "tvae", "llm_small"}
    assert planned["ctgan"].scenarios == planned["tvae"].scenarios == {2}
    assert planned["llm_small"].models["small"] == "llama3.1:8b"


def test_unknown_plan_approach_is_an_error():
    from synthgen.orchestration.run_experiments import plan_approaches

    with pytest.raises(ValueError, match="tvae2"):
        plan_approaches({"approaches": [{"name": "tvae2"}]}, {"models": {}})


def test_unapproved_profile_is_refused(capsys):
    from synthgen.orchestration.run_experiments import main

    if not (ROOT / "data" / "reference" / "reference.csv").exists():
        pytest.skip("Reference stores not built (make reference)")
    code = main(["--profile", "examples/civil_claims_profile.yaml", "--rows", "10"])
    assert code == 1 and "not approved" in capsys.readouterr().out


def test_experiment_runner_end_to_end(tmp_path):
    """Two quick approaches, two seeds, Scenario 1: batch files, report and tests."""
    from synthgen.orchestration.run_experiments import main

    if not (ROOT / "data" / "reference" / "reference.csv").exists():
        pytest.skip("Reference stores not built (run make reference)")
    # The plan's own approved profile is committed under config/, so this runs on a clone
    spec = yaml.safe_load((ROOT / "config" / "experiment_plan.yaml").read_text())["profile"]
    code = main(
        ["--profile", spec, "--scenarios", "1", "--approaches", "knowledge_based",
         "gaussian_copula", "--seeds", "1", "2", "--rows", "200", "--out", str(tmp_path)]
    )
    assert code == 0
    (folder,) = tmp_path.iterdir()
    batch = yaml.safe_load((folder / "batch.yaml").read_text())
    assert len(batch["records"]) == 4 and batch["scenario"] == 1
    assert (folder / "profile.yaml").exists()
    report = folder / "experiment_report"
    runs = pd.read_csv(report / "runs.csv")
    assert len(runs) == 4 and set(runs["seed"]) == {1, 2}
    significance = pd.read_csv(report / "significance.csv")
    assert set(significance["approach_a"]) == {"Knowledge-based"}
    assert (report / "report.html").exists()
