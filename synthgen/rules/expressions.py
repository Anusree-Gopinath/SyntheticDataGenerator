"""Safe rule language.

Rules are short expressions such as ``hearing_date > issue_date`` or
``hearing_date is not null``. They are parsed into a restricted expression tree and
evaluated as vectorised pandas operations. Rules are never run with ``eval``, so a rule
string cannot execute arbitrary code.

Supported:
- column names, numbers, quoted strings, ``null``; a column name that isn't a plain word
  (for example one with spaces) goes in backticks: ``unique(`Case Reference`)``
- comparisons: ``== != < <= > >=``
- ``is null`` / ``is not null``
- ``in [...]`` / ``not in [...]``
- ``and`` / ``or`` / ``not`` and parentheses
- ``unique(column)``
"""

from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Callable

import pandas as pd

Evaluator = Callable[[pd.DataFrame], pd.Series]


class RuleSyntaxError(ValueError):
    """Raised when a rule string uses anything outside the allowed grammar."""


_NULL_RE = re.compile(r"\bis\s+not\s+null\b|\bis\s+null\b", re.IGNORECASE)
# Quoted strings are matched first so a backtick inside 'text' is left alone
_QUOTED_NAME = re.compile(r"'[^']*'|\"[^\"]*\"|`([^`]+)`")
_ENCODED = "__col__"
_RESERVED = {"null", "and", "or", "not", "in", "is", "unique", "None", "True", "False"}


def quote_name(name: str) -> str:
    """A column name as written in a rule: backticks unless it is a plain word."""
    plain = name.isidentifier() and not keyword.iskeyword(name) and name not in _RESERVED
    return name if plain and not name.startswith("_") else f"`{name}`"


def _preprocess(expr: str) -> str:
    """Translate the rule language into a Python-shaped string for parsing only.

    Backticked column names become placeholder identifiers; ``parse_rule`` turns them back.
    """

    def name(match: re.Match[str]) -> str:
        if match.group(1) is None:
            return match.group(0)
        return _ENCODED + match.group(1).encode().hex()

    def null(match: re.Match[str]) -> str:
        return ".__notnull__()" if "not" in match.group(0).lower() else ".__isnull__()"

    text = _QUOTED_NAME.sub(name, expr)
    text = _NULL_RE.sub(null, text)
    text = re.sub(r"\bnull\b", "None", text)
    return text


def decode_names(tree: ast.AST) -> ast.AST:
    """Restore backticked column names in a parsed rule."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith(_ENCODED):
            node.id = bytes.fromhex(node.id[len(_ENCODED) :]).decode()
    return tree


def parse_rule(expr: str) -> ast.Expression:
    """Parse a rule into a Python expression tree with real column names.

    Raises ``SyntaxError`` if the text is not even shaped like an expression.
    """
    return decode_names(ast.parse(_preprocess(expr), mode="eval"))


_COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


def compile_expression(expr: str) -> Evaluator:
    """Compile a rule expression into a function that returns a boolean Series."""
    try:
        tree = parse_rule(expr)
    except SyntaxError as exc:  # pragma: no cover - message passthrough
        raise RuleSyntaxError(f"Cannot parse rule: {expr!r}") from exc
    return _compile(tree.body, expr)


def _compile(node: ast.AST, source: str) -> Evaluator:
    if isinstance(node, ast.BoolOp):
        parts = [_compile(v, source) for v in node.values]
        if isinstance(node.op, ast.And):
            return lambda df: _all(df, parts)
        return lambda df: _any(df, parts)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _compile(node.operand, source)
        return lambda df: ~inner(df).astype(bool)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise RuleSyntaxError(f"Chained comparisons are not allowed: {source!r}")
        left = _value(node.left, source)
        op = node.ops[0]
        right_node = node.comparators[0]
        if isinstance(op, (ast.In, ast.NotIn)):
            options = _literal_list(right_node, source)
            negate = isinstance(op, ast.NotIn)
            return lambda df: _membership(left(df), options, negate)
        if type(op) not in _COMPARE_OPS:
            raise RuleSyntaxError(f"Operator not allowed in rule: {source!r}")
        right = _value(right_node, source)
        fn = _COMPARE_OPS[type(op)]
        return lambda df: _compare(fn, left(df), right(df))

    if isinstance(node, ast.Call):
        # x.__isnull__() / x.__notnull__() produced by preprocessing
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"__isnull__", "__notnull__"}:
            if node.args or node.keywords:
                raise RuleSyntaxError(f"Invalid null check: {source!r}")
            col = _value(node.func.value, source)
            if node.func.attr == "__isnull__":
                return lambda df: _as_series(df, col(df)).isna()
            return lambda df: _as_series(df, col(df)).notna()
        # unique(column)
        if isinstance(node.func, ast.Name) and node.func.id == "unique" and len(node.args) == 1:
            col = _value(node.args[0], source)
            return lambda df: ~_as_series(df, col(df)).duplicated(keep=False)
        raise RuleSyntaxError(f"Function calls are not allowed in rules: {source!r}")

    raise RuleSyntaxError(f"Unsupported construct in rule: {source!r}")


def _value(node: ast.AST, source: str) -> Callable[[pd.DataFrame], object]:
    if isinstance(node, ast.Name):
        name = node.id
        if name.startswith("_"):
            raise RuleSyntaxError(f"Invalid column name in rule: {source!r}")

        def get(df: pd.DataFrame) -> pd.Series:
            if name not in df.columns:
                raise KeyError(f"Rule refers to unknown column '{name}'")
            return df[name]

        return get
    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (int, float, str))
    ):
        value = node.value
        return lambda df: value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)):
            value = -inner.value
            return lambda df: value
    if isinstance(node, (ast.BoolOp, ast.Compare, ast.UnaryOp, ast.Call)):
        return _compile(node, source)
    raise RuleSyntaxError(f"Unsupported value in rule: {source!r}")


def _literal_list(node: ast.AST, source: str) -> list[object]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise RuleSyntaxError(f"'in' needs a list of values: {source!r}")
    values = []
    for elt in node.elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float, str))):
            raise RuleSyntaxError(f"List values must be literals: {source!r}")
        values.append(elt.value)
    return values


def _as_series(df: pd.DataFrame, value: object) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(df), index=df.index)


def _compare(fn: Callable, left: object, right: object) -> pd.Series:
    result = fn(left, right)
    if isinstance(result, pd.Series):
        return result.fillna(False).astype(bool)
    raise RuleSyntaxError("A comparison must involve at least one column")


def _membership(values: object, options: list[object], negate: bool) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise RuleSyntaxError("'in' must be applied to a column")
    result = values.isin(options)
    return ~result if negate else result


def _all(df: pd.DataFrame, parts: list[Evaluator]) -> pd.Series:
    out = pd.Series(True, index=df.index)
    for part in parts:
        out &= part(df).astype(bool)
    return out


def _any(df: pd.DataFrame, parts: list[Evaluator]) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for part in parts:
        out |= part(df).astype(bool)
    return out
