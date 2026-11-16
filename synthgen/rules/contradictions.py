"""Contradiction checks run before a profile can be approved."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from synthgen.core.models import SchemaProfile
from synthgen.generators.base import CalibrationStats
from synthgen.generators.dependencies import columns_in
from synthgen.generators.knowledge_based import KnowledgeBasedGenerator
from synthgen.rules.expressions import RuleSyntaxError, compile_expression


def find_simple_conflicts(profile: SchemaProfile) -> list[str]:
    """Find conflicts that can be spotted by comparing bounds directly."""
    problems: list[str] = []
    for col in profile.columns:
        if col.min is not None and col.max is not None:
            try:
                if float(col.min) > float(col.max):
                    problems.append(f"Column '{col.name}': min is greater than max")
            except (TypeError, ValueError):
                if str(col.min) > str(col.max):
                    problems.append(f"Column '{col.name}': min is after max")
        if col.allowed_values is not None and len(col.allowed_values) == 0:
            problems.append(f"Column '{col.name}': allowed values list is empty")
    return problems


# Share of failing rows that marks a rule the generator builds rows for as conflicting
CONFLICT_SHARE = 0.01


@dataclass
class TrialFinding:
    """How one approved rule fared in the trial rows it applies to."""

    rule_id: str
    description: str
    rows_covered: int
    rows_failed: int
    used_in_generation: bool
    related: list[str] = field(default_factory=list)  # other rules on the same columns

    @property
    def fail_share(self) -> float:
        return self.rows_failed / self.rows_covered if self.rows_covered else 0.0


@dataclass
class TrialResult:
    n_rows: int
    rows_meeting_all: int  # trial rows that meet every approved rule at once
    findings: list[TrialFinding] = field(default_factory=list)
    max_fail_share: float = 0.5
    error: str | None = None

    @property
    def problems(self) -> list[TrialFinding]:
        """Rules that fail more than they should.

        A rule the generator builds rows for should hold in (nearly) every row, so more
        than ``CONFLICT_SHARE`` failing points to a conflict. A rule it only checks is
        flagged above ``max_fail_share``, since random rows can miss it by chance.
        """
        return [
            f
            for f in self.findings
            if f.rows_covered
            and f.fail_share > (CONFLICT_SHARE if f.used_in_generation else self.max_fail_share)
        ]


def trial_generation_check(
    profile: SchemaProfile,
    calibration: CalibrationStats | None = None,
    n_rows: int = 1000,
    max_fail_share: float = 0.5,
    seed: int = 0,
) -> TrialResult:
    """Generate trial rows with the knowledge-based generator and report rules that fail most.

    The generator builds rows to meet every rule it can read (see
    ``generators/dependencies.py``), so a readable rule that still fails usually
    contradicts another rule (listed in ``related``) or a column's settings. A rule the generator can't read is
    only checked, so a high failure share there may just mean random rows rarely meet it.
    Column checks (such as patterns) are left out: they are about the generator's values,
    not about rules contradicting each other.
    """
    generator = KnowledgeBasedGenerator()
    try:
        generator.fit(profile, calibration or CalibrationStats())
        rows = generator.sample(n_rows, seed=seed)
    except (KeyError, ValueError, TypeError, RuleSyntaxError) as exc:
        return TrialResult(n_rows, 0, max_fail_share=max_fail_share, error=str(exc))

    unused = set(generator.plan.unused)
    columns = {r.id: columns_in(r.expr) for r in profile.rules}
    meets_all = pd.Series(True, index=rows.index)
    findings = []
    for rule in profile.rules:
        try:
            holds = compile_expression(rule.expr)(rows)
            applies = (
                compile_expression(rule.when)(rows)
                if rule.when
                else pd.Series(True, index=rows.index)
            )
        except (KeyError, TypeError, RuleSyntaxError) as exc:
            return TrialResult(
                n_rows, 0, max_fail_share=max_fail_share, error=f"Rule {rule.id}: {exc}"
            )
        failed = applies & ~holds
        meets_all &= ~failed
        findings.append(
            TrialFinding(
                rule.id,
                rule.description,
                rows_covered=int(applies.sum()),
                rows_failed=int(failed.sum()),
                used_in_generation=rule.id not in unused,
                related=[
                    other.id
                    for other in profile.rules
                    if other.id != rule.id
                    and columns[rule.id] & (columns[other.id] | columns_in(other.when))
                ],
            )
        )
    return TrialResult(n_rows, int(meets_all.sum()), findings, max_fail_share)
