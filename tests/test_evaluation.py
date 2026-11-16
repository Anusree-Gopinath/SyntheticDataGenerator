import numpy as np
import pandas as pd
import pytest

from synthgen.evaluation.privacy import distance_to_closest_record, privacy_threshold
from synthgen.evaluation.realism import js_distance, ks_distance, proportion_gap
from synthgen.evaluation.report import mean_ci


def test_ks_identical_is_zero():
    s = pd.Series(np.arange(100.0))
    assert ks_distance(s, s) == 0.0


def test_ks_known_value():
    # Disjoint ranges: KS statistic is 1
    assert ks_distance(pd.Series([1.0, 2.0]), pd.Series([10.0, 11.0])) == 1.0


def test_js_bounds():
    assert js_distance(pd.Series(["a", "b"]), pd.Series(["a", "b"])) == pytest.approx(0.0)
    assert js_distance(pd.Series(["a"]), pd.Series(["b"])) == pytest.approx(1.0)


def test_proportion_gap():
    s = pd.Series(["a"] * 7 + ["b"] * 3)
    assert proportion_gap(s, {"a": 0.5, "b": 0.5}) == pytest.approx(0.2)


def test_exact_copy_has_zero_distance():
    sample = pd.DataFrame({"v": [1.0, 5.0, 9.0], "c": ["x", "y", "z"]})
    synthetic = sample.iloc[[0]].copy()
    assert distance_to_closest_record(synthetic, sample, ["v"])[0] == 0.0
    assert privacy_threshold(sample, ["v"]) > 0


def test_mean_ci():
    mean, lo, hi = mean_ci([1.0, 2.0, 3.0])
    assert mean == 2.0 and lo < 2.0 < hi


def test_roc_auc_and_attacks():
    from synthgen.evaluation.privacy import distance_membership_attack, roc_auc

    assert roc_auc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5
    # Synthetic data that copies the members makes them easy to spot
    members = pd.DataFrame({"v": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    others = pd.DataFrame({"v": [50.0, 60.0, 70.0], "c": ["x", "y", "z"]})
    pool = pd.concat([members.assign(member=True), others.assign(member=False)])
    result = distance_membership_attack(members, pool, ["v"])
    assert result["auc"] == 1.0
    assert result["accuracy"] == 1.0


def test_held_back_gap_is_zero_on_the_reference_world():
    from pathlib import Path

    import yaml

    from synthgen.evaluation.realism import held_back_gaps

    root = Path(__file__).resolve().parents[1]
    reference = pd.read_csv(root / "data" / "reference" / "reference.csv", dtype=str)
    held = yaml.safe_load((root / "data" / "evaluation" / "held_back_stats.yaml").read_text())
    gaps = held_back_gaps(reference, held)
    assert len(gaps) == 7
    assert max(gaps.values()) < 0.01  # held-back statistics were computed from this data
    only_two = held_back_gaps(reference[["claim_type", "case_status"]], held)
    assert list(only_two) == ["status_by_claim_type"]


def test_naive_bayes_learns_a_clear_pattern():
    from synthgen.evaluation.realism import tstr_accuracy

    train = pd.DataFrame({"x": ["a", "b"] * 50, "y": ["yes", "no"] * 50})
    assert tstr_accuracy(train, train, "y", ["x"]) == 1.0


def test_build_report_mean_and_ci():
    from synthgen.evaluation.report import build_report

    runs = pd.DataFrame({"approach": ["A", "A", "A", "B"], "gap": [1.0, 2.0, 3.0, 5.0]})
    summary = build_report(runs).set_index("approach")
    assert summary.loc["A", "mean"] == 2.0 and summary.loc["A", "n"] == 3
    assert summary.loc["A", "lo"] < 2.0 < summary.loc["A", "hi"]
    assert summary.loc["B", "lo"] == summary.loc["B", "hi"] == 5.0


def test_gower_treats_shared_missing_values_as_equal():
    from synthgen.evaluation.privacy import gower_distances

    a = pd.DataFrame({"v": [np.nan, 1.0], "c": [None, "x"]})
    b = pd.DataFrame({"v": [np.nan, np.nan], "c": [None, None]})
    d = gower_distances(a, b, ["v"])
    assert d[0, 0] == 0.0  # both missing in both columns
    assert d[1, 0] == 1.0  # present vs missing in both columns


def test_naive_bayes_bins_parsed_and_text_dates_alike():
    from synthgen.evaluation.realism import tstr_accuracy

    days = pd.date_range("2024-01-01", periods=100, freq="D")
    label = ["early"] * 50 + ["late"] * 50
    train = pd.DataFrame({"d": days, "y": label})  # parsed dates
    test = pd.DataFrame({"d": days.strftime("%Y-%m-%d"), "y": label})  # text dates
    train.loc[::10, "d"] = pd.NaT
    assert tstr_accuracy(train, test, "y", ["d"]) == 1.0
