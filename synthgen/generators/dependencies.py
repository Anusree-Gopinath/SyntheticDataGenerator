"""Read approved rules as generation steps for the knowledge-based generator.

The validator only checks rules. This module reads the same rules as instructions, so the
knowledge-based generator can build rows that meet them instead of relying on rejection:

- ``later > earlier`` between two date columns: draw ``later`` as ``earlier`` plus a gap.
- ``smaller <= larger`` between two number columns: cap ``smaller`` at ``larger``.
- ``column is not null`` / ``column is null`` with a ``when``: fill or empty ``column``
  in the rows the condition picks out (for example, listed cases have a hearing date).
- ``column == 'x'``, ``in [...]``, ``!=``, ``not in [...]`` with a ``when``: redraw
  ``column`` from the allowed values in those rows.
- ``column >= 'value'`` / ``<=`` / ``>`` / ``<`` against a fixed date or number: redraw
  ``column`` within the bounds in those rows (parts joined by ``and`` are combined).
- ``unique(column)``: the column is drawn without repeats.
- ``a and b``: each part on its own.
- ``a or b``: in rows where neither holds, one alternative is made to hold (for example
  ``a`` is applied where ``not b``). It prefers an alternative that changes a column no
  other rule changes, then the first written, as long as it fits the column order.
- A fixed value may be written on either side (``'2022-01-01' <= issue_date``).

Columns are generated in an order where everything a rule depends on comes first. A rule
that doesn't fit these shapes, or that would make the order circular, is listed in
``unused`` and left to the validator (rows that break it are rejected as before).
"""

from __future__ import annotations

import ast
import graphlib
from dataclasses import dataclass, field

from synthgen.core.models import RuleSpec, SchemaProfile
from synthgen.rules.expressions import _preprocess, decode_names, parse_rule

_ORDER_OPS = {ast.Gt: (">", True), ast.GtE: (">=", False), ast.Lt: ("<", True), ast.LtE: ("<=", False)}


@dataclass
class Order:
    """``high`` must be after (dates) or at least (numbers) ``low``.

    Without ``when`` it holds in every row: a later date is drawn from its earlier one. With
    ``when`` (from an ``or`` rule) it is applied only in the rows ``when`` picks out.
    """

    high: str
    low: str
    strict: bool
    kind: str  # "date" or "number"
    rule_id: str
    when: str | None = None

    @property
    def target(self) -> str:
        """The column changed to meet the order: the later date, or the smaller number."""
        return self.high if self.kind == "date" else self.low


@dataclass
class Presence:
    """``column`` must be present (or empty) in the rows ``when`` picks out (all if None)."""

    column: str
    present: bool
    when: str | None
    rule_id: str


@dataclass
class ValueLimit:
    """Where ``when`` holds, ``column`` must be one of ``allowed`` (or none of ``excluded``),
    and within ``low``/``high`` (``strict`` if the bound itself is not allowed)."""

    column: str
    allowed: list | None
    excluded: list
    when: str | None
    rule_id: str
    low: float | str | None = None
    high: float | str | None = None
    strict_low: bool = False
    strict_high: bool = False


@dataclass
class Unique:
    column: str
    rule_id: str


@dataclass
class DependencyPlan:
    order: list[str]  # every profile column, parents before the columns that depend on them
    orders: list[Order] = field(default_factory=list)
    presence: list[Presence] = field(default_factory=list)
    limits: list[ValueLimit] = field(default_factory=list)
    unique: set[str] = field(default_factory=set)
    unused: list[str] = field(default_factory=list)  # rule ids left to the validator

    def date_parent(self, column: str) -> Order | None:
        """The date this column is drawn after (the latest-placed one if there are several)."""
        options = [o for o in self.orders if o.kind == "date" and o.high == column and not o.when]
        return max(options, key=lambda o: self.order.index(o.low)) if options else None


def build_plan(profile: SchemaProfile, extra_edges: list[tuple[str, str]] = ()) -> DependencyPlan:
    """Turn the profile's rules into generation steps and a column order.

    ``extra_edges`` are (parent, child) pairs known from elsewhere, such as calibration
    gaps between dates or conditional shares; they are added before the rules.
    """
    names = [c.name for c in profile.columns]
    types = {c.name: c.type for c in profile.columns}
    graph: dict[str, set[str]] = {n: set() for n in names}
    plan = DependencyPlan(order=[])

    def add_edges(edges: list[tuple[str, str]]) -> bool:
        """Add parent -> child edges unless they would make the order circular."""
        trial = {k: set(v) for k, v in graph.items()}
        for parent, child in edges:
            if parent != child and parent in trial and child in trial:
                trial[child].add(parent)
        try:
            graphlib.TopologicalSorter(trial).prepare()
        except graphlib.CycleError:
            return False
        graph.clear()
        graph.update(trial)
        return True

    for edge in extra_edges:
        add_edges([edge])

    readings = {rule.id: _read_rule(rule, types) for rule in profile.rules}
    # Columns that rules with only one way to be met will change. An "or" rule prefers an
    # alternative that changes a column no other rule has claimed, so rules don't undo
    # each other; ties keep the order the alternatives are written in.
    claimed = {c for ways in readings.values() if len(ways) == 1 for s in ways[0] for c in _targets(s)}
    for rule in profile.rules:
        ways = readings[rule.id]
        if len(ways) > 1:
            ways = sorted(ways, key=lambda w: sum(c in claimed for s in w for c in _targets(s)))
        for steps in ways:
            if add_edges([edge for step in steps for edge in _edges(step)]):
                break
        else:
            plan.unused.append(rule.id)
            continue
        claimed |= {c for s in steps for c in _targets(s)}
        for step in steps:
            if isinstance(step, Order):
                plan.orders.append(step)
            elif isinstance(step, Presence):
                plan.presence.append(step)
            elif isinstance(step, Unique):
                plan.unique.add(step.column)
            else:
                plan.limits.append(step)

    # Keep the profile's column order wherever the dependencies allow it
    sorter = graphlib.TopologicalSorter(graph)
    sorter.prepare()
    rank = {n: i for i, n in enumerate(names)}
    while sorter.is_active():
        ready = sorted(sorter.get_ready(), key=rank.__getitem__)
        plan.order.extend(ready)
        sorter.done(*ready)
    return plan


def _targets(step) -> set[str]:
    """Columns a step changes."""
    if isinstance(step, Unique):
        return set()
    return {step.target} if isinstance(step, Order) else {step.column}


def _edges(step) -> list[tuple[str, str]]:
    if isinstance(step, Unique):
        return []
    if isinstance(step, Order):
        # Dates: the later one is drawn from the earlier. Numbers: the smaller is capped.
        base = [(step.low, step.high)] if step.kind == "date" else [(step.high, step.low)]
        return base + [(c, step.target) for c in columns_in(step.when) if c != step.target]
    return [(name, step.column) for name in columns_in(step.when) if name != step.column]


def columns_in(expr: str | None) -> set[str]:
    """Column names an expression mentions (none if it can't be parsed)."""
    if not expr:
        return set()
    try:
        tree = parse_rule(expr)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id != "unique"}


def _read_rule(rule: RuleSpec, types: dict[str, str]) -> list[list]:
    """Ways to meet a rule, each a list of generation steps; [] if it isn't a readable shape.

    An ordinary rule has one way. ``a or b or c`` has one per readable alternative: apply
    ``a`` where the rule's condition holds and neither ``b`` nor ``c`` does, and so on.
    """
    # Alternatives are cut from the parsed text (names still as placeholders), so their
    # conditions parse again the same way
    text = _preprocess(rule.expr)
    try:
        tree = decode_names(ast.parse(text, mode="eval")).body
    except SyntaxError:
        return []
    if not (isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.Or)):
        steps = _read_node(tree, rule.when, rule.id, types)
        return [steps] if steps else []
    ways = []
    for i, option in enumerate(tree.values):
        others = [ast.get_source_segment(text, o) for i2, o in enumerate(tree.values) if i2 != i]
        when = f"not ({' or '.join(others)})"
        if rule.when:
            when = f"({rule.when}) and {when}"
        steps = _read_node(option, when, rule.id, types, conditional=True)
        if steps:
            ways.append(steps)
    return ways


def _read_node(
    node: ast.AST, when: str | None, rule_id: str, types: dict[str, str], conditional=False
) -> list:
    """Steps for one expression (its ``and`` parts each read on its own), or []."""
    parts = node.values if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) else [node]
    steps = []
    for part in parts:
        step = _read_part(part, when, rule_id, types)
        if step is None:
            return []
        if isinstance(step, Order) and conditional:
            step.when = when
        if isinstance(step, Unique) and conditional:
            return []  # "unique or ..." has no row-by-row meaning
        steps.append(step)
    return _merge_bounds(steps)


def _merge_bounds(steps: list) -> list:
    """Join bounds on the same column (``x >= a and x <= b``) into one step."""
    out: list = []
    bounds: dict[str, ValueLimit] = {}
    for step in steps:
        is_bound = isinstance(step, ValueLimit) and (step.low is not None or step.high is not None)
        if not is_bound:
            out.append(step)
        elif step.column in bounds:
            joined = bounds[step.column]
            if step.low is not None:
                joined.low, joined.strict_low = step.low, step.strict_low
            if step.high is not None:
                joined.high, joined.strict_high = step.high, step.strict_high
        else:
            bounds[step.column] = step
            out.append(step)
    return out


_FLIP = {ast.Gt: ast.Lt, ast.Lt: ast.Gt, ast.GtE: ast.LtE, ast.LtE: ast.GtE}


def _read_part(node: ast.AST, when: str | None, rule_id: str, types: dict[str, str]):
    # unique(column)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "unique"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in types
    ):
        return Unique(node.args[0].id, rule_id)

    # column is null / column is not null
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"__isnull__", "__notnull__"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in types
    ):
        return Presence(node.func.value.id, node.func.attr == "__notnull__", when, rule_id)

    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        return None
    left, op, right = node.left, node.ops[0], node.comparators[0]
    if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
        # 'value' <= column reads as column >= 'value'
        left, right = right, left
        op = _FLIP.get(type(op), type(op))()

    # column op column: an order between two dates or two numbers
    if isinstance(left, ast.Name) and isinstance(right, ast.Name) and type(op) in _ORDER_OPS:
        a, b = left.id, right.id
        kinds = {_kind(types.get(a)), _kind(types.get(b))}
        if len(kinds) != 1 or None in kinds:
            return None
        symbol, strict = _ORDER_OPS[type(op)]
        high, low = (a, b) if symbol.startswith(">") else (b, a)
        # A date order only holds where both dates exist, so "when" adds nothing to it
        return Order(high, low, strict, kinds.pop(), rule_id)

    # column compared with a fixed date or number: a bound
    if (
        isinstance(left, ast.Name)
        and _kind(types.get(left.id))
        and isinstance(right, ast.Constant)
        and isinstance(right.value, (int, float, str))
        and type(op) in _ORDER_OPS
    ):
        symbol, strict = _ORDER_OPS[type(op)]
        limit = ValueLimit(left.id, None, [], when, rule_id)
        if symbol.startswith(">"):
            limit.low, limit.strict_low = right.value, strict
        else:
            limit.high, limit.strict_high = right.value, strict
        return limit

    # column == / != / in / not in literal values
    if isinstance(left, ast.Name) and left.id in types:
        if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant):
            values = [right.value]
        elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.List, ast.Tuple)):
            if not all(isinstance(e, ast.Constant) for e in right.elts):
                return None
            values = [e.value for e in right.elts]
        else:
            return None
        if any(v is None for v in values):
            return None
        if isinstance(op, (ast.Eq, ast.In)):
            return ValueLimit(left.id, values, [], when, rule_id)
        return ValueLimit(left.id, None, values, when, rule_id)
    return None


def _kind(column_type: str | None) -> str | None:
    if column_type == "date":
        return "date"
    if column_type in {"integer", "decimal"}:
        return "number"
    return None
