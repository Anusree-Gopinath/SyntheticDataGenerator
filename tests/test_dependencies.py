"""The knowledge-based generator reading approved rules as dependencies."""

import pandas as pd

from synthgen.core.models import ColumnSpec, RuleSpec, SchemaProfile
from synthgen.generators import CalibrationStats, KnowledgeBasedGenerator
from synthgen.generators.dependencies import build_plan
from synthgen.rules import validate


def _rule(rid, expr, when=None, rtype="conditional_value"):
    return RuleSpec(id=rid, type=rtype, description=expr, expr=expr, when=when)


def _profile(rules, extra_columns=()):
    columns = [
        ColumnSpec("case_ref", concept="case_reference", type="string", unique=True),
        ColumnSpec("claim_type", type="category", allowed_values=["Money", "Possession", "Injury"]),
        ColumnSpec("claim_value", type="decimal", min=1, max=100_000, decimals=2),
        ColumnSpec("paid_amount", type="decimal", min=0, max=100_000, decimals=2),
        ColumnSpec("status", type="category", allowed_values=["Issued", "Listed", "Closed"]),
        ColumnSpec("track", type="category", allowed_values=["Small", "Fast", "Multi"], nullable=True),
        ColumnSpec("issue_date", type="date"),
        ColumnSpec("hearing_date", type="date", nullable=True),
        ColumnSpec("judgment_date", type="date", nullable=True),
        *extra_columns,
    ]
    return SchemaProfile(profile_id="dep-test", columns=columns, rules=rules)


def _calibration():
    return CalibrationStats(
        categorical={
            "claim_type": {"Money": 0.7, "Possession": 0.2, "Injury": 0.1, "Other": 0.5},
            "track": {"Small": 0.6, "Fast": 0.3, "Multi": 0.1},
        },
        numeric={"claim_value": {"mean": 7, "std": 1, "min": 1, "max": 100_000, "dist": "lognormal"}},
        null_rate={"hearing_date": 0.8, "judgment_date": 0.5, "track": 0.5},
        pairwise={("claim_type", "defended"): {"Money": 0.1, "Injury": 0.9}},
    )


RULES = [
    _rule("R1", "hearing_date > issue_date", "hearing_date is not null", "temporal_order"),
    _rule("R2", "judgment_date >= hearing_date", "judgment_date is not null", "temporal_order"),
    _rule("R3", "hearing_date is not null", "status == 'Listed'", "conditional_presence"),
    # A condition may mention the column itself
    _rule("R4", "track in ['Fast', 'Multi']", "claim_type == 'Injury' and track is not null"),
    _rule("R5", "paid_amount <= claim_value", rtype="range"),
    _rule("R6", "track is null", "status == 'Issued'", "conditional_presence"),
]


def test_plan_reads_rules_and_orders_parents_first():
    plan = build_plan(_profile(RULES))
    assert plan.unused == []
    pos = plan.order.index
    assert pos("issue_date") < pos("hearing_date") < pos("judgment_date")
    assert pos("status") < pos("hearing_date")  # R3's condition comes first
    assert pos("claim_type") < pos("track") and pos("status") < pos("track")
    assert pos("claim_value") < pos("paid_amount")
    assert plan.date_parent("judgment_date").low == "hearing_date"


def test_plan_leaves_unreadable_and_circular_rules_to_the_validator():
    rules = [
        _rule("A", "hearing_date > issue_date", rtype="temporal_order"),
        _rule("B", "issue_date > hearing_date", rtype="temporal_order"),  # circular with A
        _rule("C", "not (status == 'Closed')"),  # 'not' isn't read
        _rule("D", "claim_type != status", rtype="conditional_value"),  # two text columns
    ]
    assert build_plan(_profile(rules)).unused == ["B", "C", "D"]


def test_generator_meets_readable_rules_without_rejection():
    profile = _profile(RULES)
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(3000, seed=5)
    result = validate(data, profile)
    assert result.passed.all(), result.failures_by_rule
    # Conditions were applied, not just met by luck
    listed = data["status"] == "Listed"
    assert data.loc[listed, "hearing_date"].notna().all()
    assert data.loc[data["status"] == "Issued", "track"].isna().all()
    # Rules only change the rows they cover
    assert data.loc[~listed, "hearing_date"].isna().mean() > 0.6


def test_generator_keeps_categories_to_allowed_values():
    gen = KnowledgeBasedGenerator()
    gen.fit(_profile([]), _calibration())
    data = gen.sample(2000, seed=1)
    assert set(data["claim_type"]) <= {"Money", "Possession", "Injury"}  # "Other" dropped
    assert abs((data["claim_type"] == "Money").mean() - 0.7) < 0.04


def test_generator_draws_within_groups():
    defended = ColumnSpec("defended", type="category")
    profile = _profile([], extra_columns=[defended])
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(6000, seed=2)
    rates = (data["defended"] == "True").groupby(data["claim_type"]).mean()
    assert abs(rates["Money"] - 0.1) < 0.03
    assert abs(rates["Injury"] - 0.9) < 0.05


def test_generator_is_reproducible_with_rules():
    gen = KnowledgeBasedGenerator()
    gen.fit(_profile(RULES), _calibration())
    pd.testing.assert_frame_equal(gen.sample(300, seed=9), gen.sample(300, seed=9))


def test_contradictory_rules_still_fail_validation():
    # Listed cases need a hearing date, and no case may have one: nothing can satisfy both
    rules = [
        _rule("P", "hearing_date is not null", "status == 'Listed'", "conditional_presence"),
        _rule("Q", "hearing_date is null", rtype="conditional_presence"),
    ]
    profile = _profile(rules)
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(500, seed=3)
    failures = validate(data, profile).failures_by_rule
    assert failures["P"] > 0 and failures["Q"] == 0


def test_strict_number_order_respects_decimals():
    rules = [_rule("S", "paid_amount < claim_value", rtype="range")]
    profile = _profile(rules)
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(1000, seed=4)
    assert (data["paid_amount"].astype(float) < data["claim_value"].astype(float)).all()


def test_bounds_against_fixed_values_are_redrawn_in_the_rows_they_cover():
    rules = [
        _rule(
            "B1",
            "issue_date >= '2022-01-01' and issue_date <= '2022-12-31'",
            "status == 'Closed'",
            "range",
        ),
        _rule("B2", "claim_value > 500", "claim_type == 'Injury'", "range"),
    ]
    profile = _profile(rules)
    plan = build_plan(profile)
    assert plan.unused == [] and len(plan.limits) == 2
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(2000, seed=6)
    assert validate(data, profile).passed.all()
    closed = data["status"] == "Closed"
    assert data.loc[closed, "issue_date"].str.startswith("2022").all()
    assert not data.loc[~closed, "issue_date"].str.startswith("2022").any()  # usual range kept


def test_unique_rule_makes_a_column_unique():
    ref = ColumnSpec("payment_ref", type="string")
    profile = _profile([_rule("U", "unique(payment_ref)", rtype="uniqueness")], [ref])
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    assert gen.sample(1000, seed=1)["payment_ref"].is_unique
    assert build_plan(profile).unique == {"payment_ref"}


def test_or_rules_are_met_by_the_first_alternative_that_fits():
    rules = [
        # Cases that aren't just issued have a hearing date
        _rule("O1", "hearing_date is not null or status == 'Issued'", rtype="conditional_presence"),
        # A payment can't exceed the claim unless the case is closed
        _rule("O2", "paid_amount <= claim_value or status == 'Closed'", rtype="range"),
        # A judgment date, where there is one, follows the hearing (or there was no hearing)
        _rule(
            "O3",
            "hearing_date is null or judgment_date > hearing_date",
            "judgment_date is not null",
            "temporal_order",
        ),
        # Fixed value written on the left
        _rule("O4", "'2024-03-01' <= issue_date", rtype="range"),
    ]
    profile = _profile(rules)
    plan = build_plan(profile)
    assert plan.unused == []
    gen = KnowledgeBasedGenerator()
    gen.fit(profile, _calibration())
    data = gen.sample(3000, seed=8)
    result = validate(data, profile)
    assert result.passed.all(), result.failures_by_rule
    issued = data["status"] == "Issued"
    assert data.loc[~issued, "hearing_date"].notna().all()
    assert data.loc[issued, "hearing_date"].isna().mean() > 0.6  # left alone where allowed
    closed = data["status"] == "Closed"
    paid, claim = data["paid_amount"].astype(float), data["claim_value"].astype(float)
    assert (paid[closed] > claim[closed]).any()  # the rule doesn't touch closed cases


def test_or_rule_falls_back_to_a_later_alternative_when_the_first_would_loop():
    rules = [
        _rule("A", "hearing_date > issue_date", "hearing_date is not null", "temporal_order"),
        # First alternative would need hearing_date before issue_date: circular with A
        _rule("B", "issue_date > hearing_date or status == 'Closed'"),
    ]
    plan = build_plan(_profile(rules))
    assert plan.unused == []
    limit = next(s for s in plan.limits if s.rule_id == "B")
    assert limit.column == "status" and limit.allowed == ["Closed"]
