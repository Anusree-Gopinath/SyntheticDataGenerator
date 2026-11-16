"""LLM generator: structured records from a local Llama model, in batches.

Each prompt carries the schema profile, the approved rules and the statistics for each
column (see ``fields.py``). The model returns a batch of rows as JSON, constrained by a JSON
schema (allowed values become an enum).

Proportions are tracked across batches. Each prompt gives exact counts for the batch (for
example "Money claim ×44, Possession claim ×3") and for empty values. The counts are worked
out so the running totals land on the target proportions, so a batch that drifts is
corrected by the next one.

Identifier and unique columns (case numbers) are filled in code, not by the model: language
models repeat themselves and can echo real-looking references, and uniqueness is
bookkeeping, not realism.

In Scenario 2 a few sample rows are added as examples. This is safe because every model
runs locally behind the gateway.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd

from synthgen.core.models import ColumnSpec
from synthgen.gateway import LLMGateway
from synthgen.gateway.gateway import GatewayError, LLMRequest
from synthgen.generators.fields import EPOCH, Field, build_fields, field_quantiles, is_identifier
from synthgen.generators.knowledge_based import KnowledgeBasedGenerator

SYSTEM = (
    "You generate fictional test data for a court case management system in England and "
    "Wales. Every value must be invented: never use real people's names, addresses or real "
    "case numbers. Follow the column descriptions, the counts and the rules exactly. Reply "
    "with JSON only."
)
EXAMPLES = 5
# Rough output budget: about 12 tokens per value, at most this many tokens per reply
TOKENS_PER_VALUE = 12
REPLY_BUDGET = 6000
MAX_FAILED_BATCHES = 3
OPTIONS = {"num_ctx": 16384, "num_predict": 8192}


class LLMGenerator:
    supports_scenarios = frozenset({1, 2})

    def __init__(
        self,
        gateway: LLMGateway,
        model: str,
        batch_size: int = 50,
        temperature: float = 0.7,
    ):
        self.gateway = gateway
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.name = f"llm:{model}"
        self.profile = None
        self.calls = 0
        # Called as on_batch(rows_so_far, rows_wanted) before each model call. The app uses it
        # to show progress; because that touches the page, it is also where Streamlit can
        # stop a run (Stop, a new run, or a closed tab) between batches instead of only after
        # the whole seed.
        self.on_batch: Callable[[int, int], None] | None = None

    # -- fit -------------------------------------------------------------------------------

    def fit(self, profile, calibration, sample: pd.DataFrame | None = None) -> None:
        self.profile = profile
        self.fields = {f.name: f for f in build_fields(profile, calibration, sample)}
        self.fallback = KnowledgeBasedGenerator()
        self.fallback.fit(profile, calibration)
        self.id_columns = [c for c in profile.columns if is_identifier(c)]
        self.columns = [c for c in profile.columns if not is_identifier(c)]
        ids = {c.name for c in self.id_columns}
        # Every rule the validator will check, except those only about identifiers
        self.rules = [r for r in profile.rules if not set(_names(r.expr)) <= ids]
        self.examples = None
        if sample is not None and not sample.empty:
            shown = sample[[c.name for c in self.columns]].dropna(axis=1, how="all")
            self.examples = shown.astype(object).where(shown.notna(), None)
        per_row = TOKENS_PER_VALUE * max(1, len(self.columns))
        self.batch = max(5, min(self.batch_size, REPLY_BUDGET // per_row))

    # -- sample ----------------------------------------------------------------------------

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.profile is None:
            raise RuntimeError("Call fit() before sample()")
        rng = np.random.default_rng(seed)
        seen = {name: Counter() for name, f in self.fields.items() if f.kind == "category"}
        empty = Counter()
        rows: list[dict] = []
        batch_no = failures = 0
        while len(rows) < n:
            if self.on_batch is not None:
                self.on_batch(len(rows), n)
            size = min(self.batch, n - len(rows))
            counts = self._batch_counts(size, len(rows), seen, empty)
            prompt = self.build_prompt(size, counts, rng)
            request = LLMRequest(
                model=self.model,
                system=SYSTEM,
                prompt=prompt,
                json_schema=self.json_schema(size),
                temperature=self.temperature,
                seed=seed * 100_003 + batch_no * 101 + failures,
                options=OPTIONS,
            )
            try:
                self.calls += 1
                batch = self.parse(self.gateway.complete_json(request), size)
            except GatewayError:
                batch = []
            if not batch:
                failures += 1
                if failures >= MAX_FAILED_BATCHES:
                    raise GatewayError(
                        f"{self.model} returned no usable rows {failures} times in a row"
                    )
                continue
            failures = 0
            batch_no += 1
            for row in batch:
                for name, counter in seen.items():
                    if row.get(name) is not None:
                        counter[row[name]] += 1
                for c in self.columns:
                    if row.get(c.name) is None:
                        empty[c.name] += 1
            rows.extend(batch)

        df = pd.DataFrame(rows[:n], columns=[c.name for c in self.columns])
        for col in self.id_columns:
            df[col.name] = self.fallback.sample_column(col, len(df), rng).to_numpy()
        return df[[c.name for c in self.profile.columns]].reset_index(drop=True)

    # -- prompt ----------------------------------------------------------------------------

    def _batch_counts(self, size: int, have: int, seen: dict, empty: Counter) -> dict:
        """Exact counts for the next batch, so running totals land on the targets."""
        out: dict[str, dict] = {}
        for c in self.columns:
            f = self.fields.get(c.name)
            if f is None:
                continue
            blanks = 0
            if f.missing > 0:
                blanks = int(np.clip(round(f.missing * (have + size) - empty[c.name]), 0, size))
            entry = {"empty": blanks}
            if f.kind == "category":
                filled = size - blanks
                done = sum(seen[c.name].values())
                need = [
                    max(0.0, p * (done + filled) - seen[c.name][cat])
                    for cat, p in zip(f.categories, f.probs, strict=True)
                ]
                entry["values"] = dict(zip(f.categories, _allocate(need, filled), strict=True))
            out[c.name] = entry
        return out

    def build_prompt(self, size: int, counts: dict, rng: np.random.Generator) -> str:
        lines = [
            f'Generate exactly {size} rows of fictional data as JSON: {{"rows": [ ... ]}}.',
            "Each row is an object with these columns. Use null for an empty value.",
            "",
            "Columns:",
        ]
        for c in self.columns:
            lines.append(f"- {self._describe(c, counts.get(c.name, {}), size)}")
        if self.rules:
            lines += ["", "Rules every row must follow:"]
            for r in self.rules:
                when = f" (only when {r.when})" if r.when else ""
                lines.append(f"- {r.description}: {r.expr}{when}")
        if self.examples is not None and len(self.examples):
            picked = self.examples.sample(
                n=min(EXAMPLES, len(self.examples)), random_state=int(rng.integers(1 << 31))
            )
            lines += ["", "Example rows from a fictional sample (match their style, do not copy):"]
            lines += [json.dumps(r, default=str) for r in picked.to_dict("records")]
        lines += [
            "",
            "Vary values from row to row. Shuffle the rows; do not group them by value.",
        ]
        return "\n".join(lines)

    def _describe(self, c: ColumnSpec, count: dict, size: int) -> str:
        f = self.fields.get(c.name)
        text = f"{c.name}"
        if c.concept:
            text += f" ({c.concept.replace('_', ' ')})"
        text += ": " + self._kind_text(c, f)
        blanks = count.get("empty", 0)
        if blanks:
            text += f" Leave it null in exactly {blanks} of the {size} rows."
        elif c.nullable and f is None:
            text += " May be null."
        elif not c.nullable:
            text += " Never null."
        if "values" in count:
            shown = ", ".join(f'"{k}" ×{v}' for k, v in count["values"].items() if v)
            text += f" Use exactly: {shown}."
        return text

    def _kind_text(self, c: ColumnSpec, f: Field | None) -> str:
        if f is not None and f.kind == "category":
            return "one of " + ", ".join(f'"{v}"' for v in f.categories) + "."
        if c.allowed_values:
            return "one of " + ", ".join(f'"{v}"' for v in c.allowed_values) + "."
        if f is not None and f.kind == "number":
            p10, p25, p50, p75, p90 = field_quantiles(f, [0.1, 0.25, 0.5, 0.75, 0.9])
            unit = "whole number" if f.integer else f"number with {f.decimals} decimal places"
            return (
                f"{unit} between {_num(f.lo)} and {_num(f.hi)}; median about {_num(p50)}, "
                f"half between {_num(p25)} and {_num(p75)}, nine in ten between {_num(p10)} "
                f"and {_num(p90)}."
            )
        if f is not None and f.kind == "date":
            lo, p50, hi = field_quantiles(f, [0.0, 0.5, 1.0])
            return f"date as YYYY-MM-DD between {_day(lo)} and {_day(hi)} (median {_day(p50)})."
        if f is not None and f.kind == "gap":
            p10, p50, p90 = field_quantiles(f, [0.1, 0.5, 0.9])
            return (
                f"date as YYYY-MM-DD, about {p50:.0f} days after {f.parent} (nine in ten "
                f"between {p10:.0f} and {p90:.0f} days after)."
            )
        if c.type == "date":
            return "date as YYYY-MM-DD."
        if c.type in {"integer", "decimal"}:
            bounds = ""
            if c.min is not None or c.max is not None:
                bounds = f" between {c.min if c.min is not None else '-'} and {c.max if c.max is not None else '-'}"
            return f"{'whole number' if c.type == 'integer' else 'number'}{bounds}."
        if c.type == "boolean":
            return '"True" or "False".'
        if c.pattern:
            return f"text matching the pattern {c.pattern}."
        if c.type == "text":
            return "short, realistic free text (no real names or addresses)."
        return "short, realistic text value."

    # -- response ----------------------------------------------------------------------------

    def json_schema(self, size: int) -> dict:
        props = {}
        for c in self.columns:
            f = self.fields.get(c.name)
            options = None
            if f is not None and f.kind == "category":
                options = list(f.categories)
            elif c.allowed_values:
                options = [str(v) for v in c.allowed_values]
            elif c.type == "boolean":
                options = ["True", "False"]
            if options is not None:
                props[c.name] = {"enum": [*options, None] if c.nullable else options}
            elif c.type == "integer" or (f is not None and f.kind == "number" and f.integer):
                props[c.name] = {"type": ["integer", "null"] if c.nullable else "integer"}
            elif c.type == "decimal":
                props[c.name] = {"type": ["number", "null"] if c.nullable else "number"}
            else:
                props[c.name] = {"type": ["string", "null"] if c.nullable else "string"}
        row = {"type": "object", "properties": props, "required": list(props)}
        return {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": row, "minItems": size, "maxItems": size}
            },
            "required": ["rows"],
        }

    def parse(self, reply: dict, size: int) -> list[dict]:
        """Rows from the model's reply, with values coerced to the profile's types."""
        raw = reply.get("rows") if isinstance(reply, dict) else None
        if not isinstance(raw, list):
            return []
        rows = []
        for item in raw[:size]:
            if not isinstance(item, dict):
                continue
            rows.append({c.name: _coerce(item.get(c.name), c, self.fields.get(c.name)) for c in self.columns})
        return rows


# -- helpers --------------------------------------------------------------------------------


def _names(expr: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*", expr)


def _allocate(need: list[float], total: int) -> list[int]:
    """Split ``total`` in proportion to ``need`` (largest remainder)."""
    if total <= 0:
        return [0] * len(need)
    weights = np.array(need, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(need))
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts))[: total - counts.sum()]:
        counts[i] += 1
    return [int(v) for v in counts]


def _coerce(value, c: ColumnSpec, f: Field | None):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        value = str(value)
    is_number = c.type in {"integer", "decimal"} or (f is not None and f.kind == "number")
    if is_number:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number):
            return None
        if c.type == "integer" or (f is not None and f.integer):
            return round(float(number))
        decimals = f.decimals if f is not None else (c.decimals if c.decimals is not None else 2)
        return round(float(number), decimals)
    if c.type == "date":
        parsed = pd.to_datetime(pd.Series([str(value)]), format="ISO8601", errors="coerce").iloc[0]
        return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
    return str(value).strip()


def _num(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"


def _day(days: float) -> str:
    return (EPOCH + pd.Timedelta(days=round(days))).strftime("%Y-%m-%d")
