"""Checks that the reference world matches its published targets and that stores stay separate."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from synthgen.reference_world.simulator import simulate
from synthgen.reference_world.stores import build_candidate_pool, draw_sample, split_statistics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    cfg = yaml.safe_load((ROOT / "config" / "reference_world.yaml").read_text())
    cfg["n_cases"] = 60000
    return simulate(cfg)


def test_matches_published_shares(reference):
    assert reference["defended"].mean() == pytest.approx(0.14, abs=0.015)
    alloc = reference["track"].dropna().value_counts(normalize=True)
    assert alloc["Small claims"] == pytest.approx(0.66, abs=0.05)
    assert alloc["Fast track"] == pytest.approx(0.26, abs=0.05)


def test_matches_published_trial_times(reference):
    t = reference[reference["hearing_type"].isin(["Small claims hearing", "Trial"])]
    weeks = (pd.to_datetime(t["hearing_date"]) - pd.to_datetime(t["issue_date"])).dt.days / 7
    assert weeks[t["track"] == "Small claims"].median() == pytest.approx(41.0, abs=1.5)
    assert weeks[t["track"] != "Small claims"].median() == pytest.approx(55.9, abs=2.5)


def test_life_cycle_is_consistent(reference):
    df = reference
    hd, iss = pd.to_datetime(df["hearing_date"]), pd.to_datetime(df["issue_date"])
    jd, wd = pd.to_datetime(df["judgment_date"]), pd.to_datetime(df["warrant_issue_date"])
    assert (hd.isna() | (hd > iss)).all()
    assert (jd.isna() | (jd >= iss)).all()
    assert (wd.isna() | (wd > jd)).all()
    assert df.loc[df["case_status"] == "Listed", "hearing_date"].notna().all()
    small = df["track"] == "Small claims"
    assert (df.loc[small, "claim_amount"] <= 10000).all()
    snap = pd.Timestamp("2026-06-30")
    assert (jd.dropna() <= snap).all() and (wd.dropna() <= snap).all()


def test_calibration_has_no_held_back_patterns(reference):
    cal, held = split_statistics(reference)
    assert not set(cal) & (set(held) - {"description"})
    text = yaml.safe_dump(cal)
    for forbidden in ["by_court", "by_value_band", "spearman", "by_representation"]:
        assert forbidden not in text


def test_candidate_pool_is_balanced(reference):
    sample = draw_sample(reference, 200, seed=1)
    pool = build_candidate_pool(reference, sample, 400, seed=1)
    assert pool["member"].sum() == 200 and len(pool) == 400
    assert set(pool[pool["member"]]["case_number"]) <= set(sample["case_number"])
    assert not set(pool[~pool["member"]]["case_number"]) & set(sample["case_number"])


def test_calibration_and_sample_use_profile_column_names():
    from synthgen.core.models import ColumnSpec, SchemaProfile
    from synthgen.reference_world.stores import calibration_for_profile, sample_for_profile

    profile = SchemaProfile(
        "p",
        columns=[
            ColumnSpec("amt", concept="claim_amount", type="decimal"),
            ColumnSpec("crt", concept="court_name", type="category"),
            ColumnSpec("iss", concept="issue_date", type="date"),
            ColumnSpec("hrg", concept="hearing_date", type="date", nullable=True),
            ColumnSpec("def_flag", concept="defended_flag", type="boolean"),
            ColumnSpec("notes", type="text"),
        ],
    )
    cal = yaml.safe_load((ROOT / "data" / "calibration" / "calibration_stats.yaml").read_text())
    stats = calibration_for_profile(cal, profile)
    assert set(stats.numeric) == {"amt"}
    assert "crt" in stats.categorical and "court" not in stats.categorical
    assert stats.categorical["def_flag"]["True"] == cal["rates"]["defended"]
    assert ("iss", "hrg") in stats.gaps_days
    assert "notes" not in stats.null_rate

    sample = pd.read_csv(ROOT / "data" / "calibration" / "scenario2_sample.csv", dtype=str)
    mapped = sample_for_profile(sample, profile)
    assert list(mapped.columns) == ["amt", "crt", "iss", "hrg", "def_flag", "notes"]
    assert mapped["notes"].isna().all()
