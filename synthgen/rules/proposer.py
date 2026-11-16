"""Rule proposer: suggest cross-column rules from the dictionary and the local LLM.

Each proposed rule is checked against the example rows (if any) and shown to the user
with its evidence. Nothing is applied until the user approves it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from synthgen.core.models import RULE_TYPES, ColumnSpec, RuleSpec, SchemaProfile
from synthgen.gateway import LLMGateway, LLMRequest
from synthgen.rules.expressions import RuleSyntaxError, compile_expression, quote_name


@dataclass
class RuleEvidence:
    rule: RuleSpec
    rows_checked: int
    rows_holding: int

    @property
    def fails(self) -> bool:
        return self.rows_holding < self.rows_checked

    @property
    def summary(self) -> str:
        if self.rows_checked == 0:
            return "No matching example rows to check"
        if self.fails:
            failing = self.rows_checked - self.rows_holding
            return f"Fails in {failing} of {self.rows_checked} matching example rows"
        return f"Holds in {self.rows_holding} of {self.rows_checked} matching example rows"


def check_against_sample(rule: RuleSpec, sample: pd.DataFrame) -> RuleEvidence:
    """Check a proposed rule against the fictional example rows."""
    if sample.empty:
        return RuleEvidence(rule, 0, 0)
    rows = sample
    if rule.when:
        rows = sample[compile_expression(rule.when)(sample)]
    holds = compile_expression(rule.expr)(rows) if len(rows) else pd.Series(dtype=bool)
    return RuleEvidence(rule, rows_checked=len(rows), rows_holding=int(holds.sum()))


def rule_error(rule: RuleSpec, column_names: list[str]) -> str | None:
    """Return why a rule can't be used with these columns, or None if it is fine."""
    empty = pd.DataFrame(columns=column_names)
    try:
        for expr in (rule.expr, rule.when):
            if expr:
                compile_expression(expr)(empty)
    except (RuleSyntaxError, KeyError, TypeError) as exc:
        return str(exc).strip("\"'")
    return None


# Quoted strings are matched first so values such as 'Listed' are never renamed.
_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|\b[A-Za-z_]\w*\b")


def _concepts_to_columns(expr: str | None, concepts: set[str], mapping: dict[str, str]):
    """Rewrite concept names in a dictionary rule as column names.

    Returns None if the rule mentions a concept that no column has.
    """
    if not expr:
        return expr
    missing = False

    def repl(match: re.Match[str]) -> str:
        nonlocal missing
        token = match.group(0)
        if token in concepts:
            if token not in mapping:
                missing = True
                return token
            return quote_name(mapping[token])
        return token

    out = _TOKEN.sub(repl, expr)
    return None if missing else out


def propose_from_dictionary(columns: list[ColumnSpec], dictionary) -> list[RuleSpec]:
    """Turn each matched concept's ``typical_rules`` into candidate rules.

    Dictionary rules are written with concept names (``hearing_date > issue_date``) and
    are rewritten with this schema's column names. A rule is only proposed when every
    concept it mentions is matched to a column.
    """
    mapping: dict[str, str] = {}
    for col in columns:
        if col.concept and col.concept not in mapping:
            mapping[col.concept] = col.name
    concepts = {c.concept for c in dictionary}
    rules: list[RuleSpec] = []
    seen: set[tuple[str, str | None]] = set()
    for concept in mapping:
        for typical in dictionary.get(concept).typical_rules:
            expr = _concepts_to_columns(typical["expr"], concepts, mapping)
            when = _concepts_to_columns(typical.get("when"), concepts, mapping)
            if expr is None or (typical.get("when") and when is None) or (expr, when) in seen:
                continue
            seen.add((expr, when))
            rules.append(
                RuleSpec(
                    id=f"D{len(rules) + 1}",
                    type=typical["type"],
                    description=typical.get("description", expr),
                    expr=expr,
                    when=when,
                    source="dictionary",
                )
            )
    return rules


RULES_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": sorted(RULE_TYPES)},
                    "description": {"type": "string"},
                    "expr": {"type": "string"},
                    "when": {"type": ["string", "null"]},
                },
                "required": ["type", "description", "expr"],
            },
        }
    },
    "required": ["rules"],
}

RULES_SYSTEM_PROMPT = (
    "You propose cross-column business rules for a court case management test data schema. "
    "Write every rule in the rule language described, using only the column names given. "
    "Reply with JSON only."
)

RULE_LANGUAGE_HELP = """Rule language:
- column names, numbers, quoted strings, null
- comparisons: == != < <= > >=
- is null / is not null
- in [...] / not in [...]
- and / or / not, parentheses
- unique(column)
- a column name that isn't a single word (for example with spaces) goes in backticks: `Case Reference`
Each rule has an expr that must hold, and an optional when condition saying which rows it applies to.
Example: {"type": "conditional_presence", "description": "Listed cases have a hearing date",
"expr": "hearing_date is not null", "when": "status == 'Listed'"}"""


def build_rules_prompt(profile: SchemaProfile) -> str:
    lines = [RULE_LANGUAGE_HELP, "", "Columns:"]
    for col in profile.columns:
        line = f"- {col.name}: {col.concept or 'unknown meaning'}, {col.type}"
        if col.nullable:
            line += ", optional"
        if col.allowed_values:
            line += f", one of {col.allowed_values}"
        lines.append(line)
    return "\n".join(lines)


def propose_from_llm(
    gateway: LLMGateway, model: str, profile: SchemaProfile, seed: int = 0
) -> list[RuleSpec]:
    """Ask the local LLM for candidate rules. Rules that don't parse or that use unknown
    columns are dropped, because the user can only approve rules the engine can check."""
    request = LLMRequest(
        model=model,
        system=RULES_SYSTEM_PROMPT,
        prompt=build_rules_prompt(profile),
        json_schema=RULES_RESPONSE_SCHEMA,
        seed=seed,
    )
    data = gateway.complete_json(request)
    names = [c.name for c in profile.columns]
    rules: list[RuleSpec] = []
    for item in data.get("rules", []):
        try:
            rule = RuleSpec(
                id=f"L{len(rules) + 1}",
                type=item["type"],
                description=item["description"],
                expr=item["expr"],
                when=item.get("when") or None,
                source="llm",
            )
        except (KeyError, ValueError):
            continue
        if rule_error(rule, names) is None:
            rules.append(rule)
    return rules
