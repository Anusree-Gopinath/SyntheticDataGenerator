"""Reference world simulator (version 2).

Generates fictional county court cases from documented rules. Proportions and timings are
calibrated to published statistics; relationships between fields (for example, higher-value
claims being defended more often, or some courts being slower) are documented assumptions that
are deliberately richer than anything generators are told.

The world is observed at a snapshot date: events after it (a trial not yet heard, a warrant not
yet issued) are not recorded, just as in a real case management system.

The full dataset goes to the evaluation store only. Generators see calibration statistics.

Run: ``python -m synthgen.reference_world.simulator --config config/reference_world.yaml``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLUMNS = [
    "case_number", "claim_type", "claim_amount", "track", "issue_date", "court", "region",
    "defended", "claimant_represented", "defendant_represented", "interpreter_required",
    "hearing_date", "hearing_type", "case_status", "outcome", "judgment_type", "judgment_date",
    "judgment_amount", "warrant_issue_date",
]


def _logistic(x):
    return 1 / (1 + np.exp(-x))


def simulate(config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    n = config["n_cases"]
    snap = pd.Timestamp(config["snapshot_date"])

    # Claim type and value
    types = config["claim_types"]
    names = list(types)
    shares = np.array([types[t]["share"] for t in names], dtype=float)
    claim_type = rng.choice(names, size=n, p=shares / shares.sum())
    value = np.full(n, np.nan)
    for t in names:
        if "value_median" in types[t]:
            idx = claim_type == t
            value[idx] = types[t]["value_median"] * rng.lognormal(0, types[t]["value_sigma"], idx.sum())
            if "high_value_share" in types[t]:
                # A second, higher-value group (for example specified debts and contract claims)
                hi = idx & (rng.random(n) < types[t]["high_value_share"])
                value[hi] = types[t]["high_value_median"] * rng.lognormal(0, types[t]["high_value_sigma"], hi.sum())
    value = np.round(np.clip(value, 25, 500_000), 2)
    is_money = np.isin(claim_type, ["Money claim", "Personal injury", "Other damages"])
    is_poss = claim_type == "Possession claim"

    # Issue date and court (each court has a hidden speed factor)
    start = pd.Timestamp(config["issue_dates"]["start"])
    span = (pd.Timestamp(config["issue_dates"]["end"]) - start).days
    issue = start + pd.to_timedelta(rng.integers(0, span + 1, n), unit="D")
    courts = list(config["courts"])
    court = rng.choice(courts, size=n)
    region = np.array([config["courts"][c] for c in court])
    tt = config["time_to_trial"]
    # Court speed factors use their own random stream (so they don't change with n_cases) and are
    # centred so their geometric mean is 1
    raw = np.random.default_rng(config["seed"] + 1).lognormal(0, tt["court_factor_sigma"], len(courts))
    court_factor_map = dict(zip(courts, raw / np.exp(np.log(raw).mean())))
    court_factor = np.array([court_factor_map[c] for c in court])

    # Defence: more likely for higher-value claims (hidden relationship)
    d = config["defence"]
    p_def = _logistic(d["intercept"] + d["log_value_effect"] * np.log(np.nan_to_num(value, nan=500)))
    p_def = np.where(is_poss, d["possession_rate"], np.where(is_money, p_def, 0.10))
    defended = rng.random(n) < p_def

    # Representation
    rd = config["representation_defended"]
    combo = rng.choice(list(rd), size=n, p=np.array(list(rd.values())) / sum(rd.values()))
    ru = config["representation_undefended"]
    clmt_rep = np.where(defended, np.isin(combo, ["both", "claimant_only"]), rng.random(n) < ru["claimant"])
    def_rep = np.where(defended, np.isin(combo, ["both", "defendant_only"]), rng.random(n) < ru["defendant"])
    interpreter = rng.random(n) < config["interpreter_rate"]

    # Track allocation for defended money and damages claims (CPR 26.9 limits)
    tr = config["tracks"]
    small_limit = np.where(claim_type == "Personal injury", tr["small_claims_limit_personal_injury"], tr["small_claims_limit"])
    track = np.select(
        [value <= small_limit, value <= tr["fast_track_limit"], value <= tr["intermediate_track_limit"]],
        ["Small claims", "Fast track", "Intermediate track"],
        default="Multi-track",
    ).astype(object)
    complex_case = (value > tr["fast_track_limit"]) & (rng.random(n) < tr["complex_to_multi_share"])
    track[complex_case] = "Multi-track"
    allocated = defended & is_money
    track[~allocated] = None

    # Outcomes start empty
    hearing = pd.Series(pd.NaT, index=range(n))
    hearing_type = np.full(n, None, dtype=object)
    status = np.full(n, "Issued", dtype=object)
    outcome = np.full(n, None, dtype=object)
    jtype = np.full(n, None, dtype=object)
    jdate = pd.Series(pd.NaT, index=range(n))
    jamount = np.full(n, np.nan)

    # Defended money claims: trial or settlement
    t = config["trial"]
    neither = ~clmt_rep & ~def_rep
    both = clmt_rep & def_rep
    p_trial = _logistic(t["intercept"] + t["neither_represented_effect"] * neither + t["both_represented_effect"] * both)
    to_trial = allocated & (rng.random(n) < p_trial)
    med = np.where(track == "Small claims", tt["small_claims_median_days"], tt["other_tracks_median_days"])
    factor = court_factor * np.where(neither, tt["neither_represented_factor"], 1.0) * np.where(interpreter, tt["interpreter_factor"], 1.0)
    high = np.log(np.maximum(np.nan_to_num(value, nan=1) / tr["fast_track_limit"], 1.0))
    factor = factor * np.exp(tt["high_value_log_effect"] * high)
    days = np.round(med * factor * rng.lognormal(0, tt["sigma"], n)).astype(int)
    hearing[to_trial] = (issue + pd.to_timedelta(days, unit="D"))[to_trial]
    hearing_type[to_trial] = np.where(track[to_trial] == "Small claims", "Small claims hearing", "Trial")
    heard = to_trial & (hearing <= snap).to_numpy()
    status[to_trial & ~heard] = "Listed"
    status[heard] = "Heard"
    win = heard & (rng.random(n) < t["claimant_win_rate"])
    trial_jdate = hearing + pd.to_timedelta(rng.integers(0, 29, n), unit="D")
    win = win & (trial_jdate <= snap).to_numpy()  # judgment not yet given at the snapshot otherwise
    outcome[heard] = np.where(win[heard], "Judgment for claimant", "Dismissed")
    jtype[win] = "After trial"
    jdate[win] = trial_jdate[win]
    jamount[win] = np.round(value[win] * rng.uniform(0.4, 1.0, win.sum()), 2)

    sd = config["settlement_days"]
    settle_date = issue + pd.to_timedelta(rng.integers(sd["min"], sd["max"] + 1, n), unit="D")
    settles = allocated & ~to_trial
    settled_now = settles & (settle_date <= snap)
    status[settles & ~settled_now] = "Defended"
    status[settled_now] = "Settled"
    outcome[settled_now] = "Settled"

    # Undefended money claims: default judgment, admission, withdrawal or still open
    u = config["undefended"]
    und = is_money & ~defended
    r = rng.random(n)
    jd = issue + pd.to_timedelta(rng.integers(u["days_to_judgment"]["min"], u["days_to_judgment"]["max"] + 1, n), unit="D")
    default_j = und & (r < u["default_judgment_rate"]) & (jd <= snap)
    admission = und & (r >= u["default_judgment_rate"]) & (r < u["default_judgment_rate"] + u["admission_rate"]) & (jd <= snap)
    withdrawn = und & (r >= 1 - u["withdrawn_rate"])
    for mask, name in [(default_j, "Default"), (admission, "Admission")]:
        jtype[mask] = name
        jdate[mask] = jd[mask]
        jamount[mask] = value[mask]
        status[mask] = "Judgment"
    status[withdrawn] = "Closed"
    outcome[withdrawn] = "Withdrawn"

    # Possession claims: a possession hearing, usually an order
    p = config["possession"]
    pdays = np.round(p["hearing_median_days"] * rng.lognormal(0, 0.3, n)).astype(int)
    phear = issue + pd.to_timedelta(pdays, unit="D")
    hearing[is_poss] = phear[is_poss]
    hearing_type[is_poss] = "Possession hearing"
    p_heard = is_poss & (phear <= snap)
    status[is_poss & ~p_heard] = "Listed"
    order = p_heard & (rng.random(n) < p["order_rate"])
    status[p_heard] = "Heard"
    outcome[p_heard] = np.where(order[p_heard], "Possession order", "Dismissed")
    jtype[order] = "Possession order"
    jdate[order] = phear[order]

    # Other non-money claims: mostly closed after a while
    other = claim_type == "Other non-money"
    closed = other & (issue + pd.Timedelta(days=120) <= snap) & (rng.random(n) < 0.7)
    status[closed] = "Closed"
    outcome[closed] = "Withdrawn"

    # Enforcement: warrants after judgment (low-value claims enforced more; hidden)
    w = config["warrant"]
    has_j = pd.notna(jdate).to_numpy()
    p_w = _logistic(w["intercept"] + w["low_value_effect"] * (np.nan_to_num(value, nan=5000) < 1000) + w["defendant_represented_effect"] * def_rep)
    p_w = np.where(jtype == "Possession order", p["warrant_rate"], p_w)
    wd = jdate + pd.to_timedelta(rng.integers(w["days_after_judgment"]["min"], w["days_after_judgment"]["max"] + 1, n), unit="D")
    warrant = has_j & (rng.random(n) < p_w) & (wd <= snap).to_numpy()
    wdate = wd.where(warrant)

    def fmt(s):
        return pd.Series(s).dt.strftime("%Y-%m-%d").where(pd.notna(s), None)

    df = pd.DataFrame(
        {
            "case_number": [f"SYN{i:07d}" for i in range(n)],
            "claim_type": claim_type,
            "claim_amount": value,
            "track": track,
            "issue_date": fmt(issue),
            "court": court,
            "region": region,
            "defended": defended,
            "claimant_represented": clmt_rep,
            "defendant_represented": def_rep,
            "interpreter_required": interpreter,
            "hearing_date": fmt(hearing),
            "hearing_type": hearing_type,
            "case_status": status,
            "outcome": outcome,
            "judgment_type": jtype,
            "judgment_date": fmt(jdate),
            "judgment_amount": jamount,
            "warrant_issue_date": fmt(wdate),
        }
    )
    return df[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reference_world.yaml")
    parser.add_argument("--out", default="data/reference/reference.csv")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    df = simulate(config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} fictional cases to {args.out}")


if __name__ == "__main__":
    main()
