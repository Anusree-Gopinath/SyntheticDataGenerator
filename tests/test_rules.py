import pandas as pd
import pytest

from synthgen.rules import RuleSyntaxError, compile_expression, validate


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "issue_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "hearing_date": ["2025-02-01", "2025-01-15", None],
            "status": ["Listed", "Listed", "Listed"],
            "claim_value": [100.0, 20000.0, 5000.0],
        }
    )


def test_comparison_between_columns(df):
    result = compile_expression("hearing_date > issue_date")(df)
    assert result.tolist() == [True, False, False]


def test_null_checks(df):
    assert compile_expression("hearing_date is null")(df).tolist() == [False, False, True]
    assert compile_expression("hearing_date is not null")(df).tolist() == [True, True, False]


def test_boolean_and_membership(df):
    expr = "status in ['Listed', 'Heard'] and claim_value <= 10000"
    assert compile_expression(expr)(df).tolist() == [True, False, True]


def test_unique():
    data = pd.DataFrame({"c": ["a", "b", "a"]})
    assert compile_expression("unique(c)")(data).tolist() == [False, True, False]


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('echo hi')",
        "claim_value.__class__",
        "open('x')",
        "[x for x in claim_value]",
        "lambda: 1",
    ],
)
def test_rejects_code(malicious):
    with pytest.raises(RuleSyntaxError):
        compile_expression(malicious)


def test_validator_counts_failures(profile):
    data = pd.DataFrame(
        {
            "case_number": ["K01AB123", "K01AB124", "bad"],
            "claim_type": ["Money claim"] * 3,
            "claim_value": [100.0, 200.0, 300.0],
            "issue_date": ["2025-01-01"] * 3,
            "hearing_date": ["2025-03-01", None, "2024-12-01"],
            "court": ["Cardiff"] * 3,
            "status": ["Listed", "Listed", "Defended"],
        }
    )
    result = validate(data, profile)
    assert result.passed.tolist() == [True, False, False]
    assert result.failures_by_rule["R2"] == 1  # listed without a hearing date
    assert result.failures_by_rule["R1"] == 1  # hearing before issue
    assert result.failures_by_rule["column:case_number"] == 1


def test_dictionary_rules_use_column_names(dictionary):
    from synthgen.core.models import ColumnSpec
    from synthgen.rules.proposer import propose_from_dictionary

    columns = [
        ColumnSpec("hrg_dt", concept="hearing_date", type="date"),
        ColumnSpec("iss_dt", concept="issue_date", type="date"),
    ]
    [rule] = propose_from_dictionary(columns, dictionary)
    assert rule.expr == "hrg_dt > iss_dt"
    assert rule.when == "hrg_dt is not null"
    assert rule.source == "dictionary"
    assert not rule.approved


def test_dictionary_rule_needs_every_concept(dictionary):
    from synthgen.core.models import ColumnSpec
    from synthgen.rules.proposer import propose_from_dictionary

    columns = [ColumnSpec("hearing_date", concept="hearing_date", type="date")]
    assert propose_from_dictionary(columns, dictionary) == []


def test_dictionary_rewrite_leaves_quoted_values():
    from synthgen.rules.proposer import _concepts_to_columns

    out = _concepts_to_columns(
        "case_status == 'case_status'", {"case_status"}, {"case_status": "status"}
    )
    assert out == "status == 'case_status'"


class FakeGateway:
    def __init__(self, response):
        self.response = response

    def complete_json(self, request):
        return self.response


def test_llm_rules_dropped_when_unusable(profile):
    from synthgen.rules.proposer import propose_from_llm

    response = {
        "rules": [
            {"type": "conditional_presence", "description": "ok",
             "expr": "hearing_date is not null", "when": "status == 'Listed'"},
            {"type": "range", "description": "unknown column", "expr": "fee > 0"},
            {"type": "range", "description": "code", "expr": "__import__('os')"},
            {"type": "made_up", "description": "bad type", "expr": "claim_value > 0"},
        ]
    }
    rules = propose_from_llm(FakeGateway(response), "llama3.1:8b", profile)
    assert [r.description for r in rules] == ["ok"]
    assert rules[0].source == "llm"


def test_rule_evidence_reports_failures():
    from synthgen.core.models import RuleSpec
    from synthgen.rules.proposer import check_against_sample

    rule = RuleSpec("R1", "conditional_presence", "d", "hearing_date is not null",
                    when="status == 'Defended'")
    sample = pd.DataFrame({"hearing_date": [None, "2025-01-01"],
                           "status": ["Defended", "Listed"]})
    evidence = check_against_sample(rule, sample)
    assert evidence.fails
    assert evidence.summary == "Fails in 1 of 1 matching example rows"


# -- trial-generation check ------------------------------------------------------------


def _trial_profile(rules):
    from synthgen.core.models import ColumnSpec, RuleSpec, SchemaProfile

    columns = [
        ColumnSpec("status", type="category", allowed_values=["Issued", "Listed"]),
        ColumnSpec("issue_date", type="date"),
        ColumnSpec("hearing_date", type="date", nullable=True),
        ColumnSpec("fee", type="decimal", min=0, max=500, decimals=2),
    ]
    return SchemaProfile(
        profile_id="trial",
        columns=columns,
        rules=[RuleSpec(id=i, type=t, description=e, expr=e, when=w) for i, t, e, w in rules],
    )


def test_trial_check_passes_rules_that_fit_together(profile):
    from synthgen.rules.contradictions import trial_generation_check

    result = trial_generation_check(profile, n_rows=500)
    assert result.error is None
    assert result.problems == []
    assert result.rows_meeting_all == 500
    assert all(f.used_in_generation for f in result.findings)


def test_trial_check_flags_contradicting_rules():
    from synthgen.rules.contradictions import trial_generation_check

    profile = _trial_profile([
        ("A", "conditional_presence", "hearing_date is not null", "status == 'Listed'"),
        ("B", "conditional_presence", "hearing_date is null", None),
        ("C", "temporal_order", "hearing_date > issue_date", "hearing_date is not null"),
    ])
    result = trial_generation_check(profile, n_rows=400)
    flagged = {f.rule_id: f for f in result.problems}
    assert set(flagged) == {"A"}  # B is applied after A, so A is the one that keeps failing
    assert set(flagged["A"].related) == {"B", "C"}
    assert result.rows_meeting_all < 400


def test_trial_check_flags_a_bound_nothing_can_meet():
    from synthgen.rules.contradictions import trial_generation_check

    profile = _trial_profile([("F", "range", "fee > 600", None)])  # the column's max is 500
    result = trial_generation_check(profile, n_rows=200)
    assert [f.rule_id for f in result.problems] == ["F"]


def test_trial_check_reports_rules_with_unknown_columns():
    from synthgen.rules.contradictions import trial_generation_check

    profile = _trial_profile([("X", "range", "missing_col > 1", None)])
    result = trial_generation_check(profile, n_rows=50)
    assert result.error and "X" in result.error


def test_trial_check_flags_a_conflict_in_a_minority_of_rows():
    from synthgen.rules.contradictions import trial_generation_check

    # Listed cases must have a hearing date, but every case dated before 2025 must not:
    # only some Listed rows are affected, well under half of them
    profile = _trial_profile([
        ("A", "conditional_presence", "hearing_date is not null", "status == 'Listed'"),
        ("B", "conditional_presence", "hearing_date is null", "issue_date < '2024-04-01'"),
    ])
    result = trial_generation_check(profile, n_rows=2000)
    a = next(f for f in result.findings if f.rule_id == "A")
    assert 0.01 < a.fail_share < 0.5
    assert [f.rule_id for f in result.problems] == ["A"]


# -- column names with spaces ----------------------------------------------------------


def test_backticked_column_names_with_spaces():
    import pandas as pd

    from synthgen.rules.expressions import compile_expression, quote_name

    df = pd.DataFrame(
        {
            "Case Reference": ["A", "B", "A"],
            "Hearing Date": ["2025-02-01", None, "2025-01-01"],
            "Decision Date": ["2025-03-01", "2025-01-05", "2024-12-01"],
        }
    )
    assert compile_expression("unique(`Case Reference`)")(df).tolist() == [False, True, False]
    rule = compile_expression("`Decision Date` > `Hearing Date` or `Hearing Date` is null")
    assert rule(df).tolist() == [True, True, False]
    # Backticks inside quoted text are text, not a column
    assert compile_expression("`Case Reference` != 'a `b` c'")(df).all()
    assert quote_name("Case Reference") == "`Case Reference`"
    assert quote_name("case_ref") == "case_ref"
    assert quote_name("status") == "status" and quote_name("in") == "`in`"


def test_dictionary_rules_quote_column_names_with_spaces(dictionary):
    from synthgen.core.models import ColumnSpec, SchemaProfile
    from synthgen.generators.dependencies import build_plan, columns_in
    from synthgen.rules.proposer import propose_from_dictionary, rule_error

    columns = [
        ColumnSpec("Case Reference", concept="case_reference"),
        ColumnSpec("Issue Date", concept="issue_date", type="date"),
        ColumnSpec("Hearing Date", concept="hearing_date", type="date", nullable=True),
    ]
    rules = propose_from_dictionary(columns, dictionary)
    exprs = {r.expr for r in rules}
    assert "unique(`Case Reference`)" in exprs
    assert "`Hearing Date` > `Issue Date`" in exprs
    names = [c.name for c in columns]
    assert all(rule_error(r, names) is None for r in rules)
    assert columns_in("unique(`Case Reference`)") == {"Case Reference"}
    assert columns_in("unique(Case Reference)") == set()  # unparseable: no crash
    # The knowledge-based generator reads them too
    plan = build_plan(SchemaProfile("spaces", columns=columns, rules=rules))
    assert plan.unused == [] and plan.unique == {"Case Reference"}
    assert plan.date_parent("Hearing Date").low == "Issue Date"
