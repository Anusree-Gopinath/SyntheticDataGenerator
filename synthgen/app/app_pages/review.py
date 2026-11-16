"""Steps 2 to 4: identify columns, propose rules, review and approve.

Every inferred column and rule is a suggestion until the user accepts it. Only the
approved profile is saved (as a new version, with a change log) and used to generate data.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from synthgen.app import ui
from synthgen.core.models import RULE_TYPES, RuleSpec, SchemaProfile
from synthgen.core.profile_manager import (
    build_column,
    coerce_sample,
    next_version,
    profile_id_from_filename,
    save_profile,
)
from synthgen.gateway import GatewayError, LLMGateway
from synthgen.generators.dependencies import columns_in
from synthgen.reference_world.stores import calibration_for_profile
from synthgen.rules.contradictions import TrialResult, find_simple_conflicts, trial_generation_check
from synthgen.rules.proposer import (
    check_against_sample,
    propose_from_dictionary,
    propose_from_llm,
    rule_error,
)
from synthgen.schema import ConceptDictionary, RulesMatcher
from synthgen.schema.combine import combine
from synthgen.schema.embedding_classifier import EmbeddingClassifier, load_labelled_columns
from synthgen.schema.llm_classifier import LLMClassifier

ROOT = Path(__file__).resolve().parents[3]
PROFILE_STORE = ROOT / "data" / "profiles"
CALIBRATION_STORE = ROOT / "data" / "calibration" / "calibration_stats.yaml"
TRIAL_ROWS = 1000
TEST_SCHEMAS = ROOT / "examples" / "test_schemas"
METHOD_LABELS = {"rules": "rules", "embedding": "embedding classifier", "llm": "local LLM"}
RULE_TYPE_LABELS = {t: t.replace("_", " ").capitalize() for t in sorted(RULE_TYPES)}


@st.cache_resource
def load_dictionary() -> ConceptDictionary:
    return ConceptDictionary.load(ROOT / "config" / "concept_dictionary_extended.yaml")


@st.cache_resource
def load_settings() -> tuple[dict, float]:
    models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    thresholds = yaml.safe_load((ROOT / "config" / "thresholds.yaml").read_text())
    return models, float(thresholds["column_confidence"]["value"])


@st.cache_resource
def load_gateway(url: str) -> LLMGateway:
    return LLMGateway(url)


@st.cache_resource(show_spinner="Loading the local embedding model...")
def load_embedding(_dictionary: ConceptDictionary, model_name: str):
    """The embedding classifier, trained on the train split of the test schemas.

    Returns (classifier, None), or (None, reason) when the method can't run here.
    """
    try:
        classifier = EmbeddingClassifier(_dictionary, model_name)
        if (TEST_SCHEMAS / "index.csv").exists():
            classifier.fit(load_labelled_columns(TEST_SCHEMAS, {"train"}))
        return classifier, None
    except ImportError:
        return None, 'Install it with pip install -e ".[embeddings]" to add a third method.'
    except OSError as exc:
        return None, f"The embedding model could not be loaded: {exc}"


@st.cache_data(show_spinner=False)
def run_trial(profile_yaml: str) -> TrialResult:
    """Trial-generate rows for a draft profile (cached until the draft changes).

    Uses the calibration statistics where they exist, so empty values and dates look
    realistic; otherwise only the profile.
    """
    draft = SchemaProfile.from_dict(yaml.safe_load(profile_yaml))
    calibration = None
    if CALIBRATION_STORE.exists():
        calibration = calibration_for_profile(yaml.safe_load(CALIBRATION_STORE.read_text()), draft)
    return trial_generation_check(draft, calibration, n_rows=TRIAL_ROWS)


def code(text: str) -> str:
    """Inline code that can itself contain backticks (rules quote column names with them)."""
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def concept_label(concept: str | None) -> str:
    return concept.replace("_", " ").capitalize() if concept else "Not in dictionary"


# Bump when inference output changes shape, so open sessions infer again
INFERENCE_VERSION = "2"  # 2: column names with spaces are backticked in proposed rules


def fingerprint(schema) -> str:
    text = INFERENCE_VERSION + ",".join(schema.headers) + "\n" + schema.sample.to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def run_inference(schema, dictionary, gateway, embedding, llm_model, llm_label, threshold) -> dict:
    """Run every available method, combine them and propose rules."""
    sample = schema.sample if schema.has_rows else None
    by_column = {h: [] for h in schema.headers}
    methods, notes = ["rules"], []

    matcher = RulesMatcher(dictionary)
    for h in schema.headers:
        values = sample[h] if sample is not None else None
        by_column[h].append(matcher.infer(h, values))

    if embedding is not None:
        for inf in embedding.infer_all(schema.headers, sample):
            by_column[inf.column].append(inf)
        methods.append("embedding")

    if llm_model:
        try:
            classifier = LLMClassifier(gateway, dictionary, llm_model)
            for inf in classifier.infer_all(schema.headers, sample):
                if inf.column in by_column:
                    by_column[inf.column].append(inf)
            methods.append("llm")
        except GatewayError as exc:
            notes.append(f"The local LLM could not identify columns: {exc}")

    combined, columns, status = {}, [], {}
    for h in schema.headers:
        result = combine(by_column[h], weights={}, review_threshold=threshold)
        combined[h] = result
        inferred_by = {inf.method: inf.confidence for inf in by_column[h]}
        columns.append(build_column(h, result.concept, dictionary, sample, inferred_by))
        status[h] = "review" if result.needs_review else "accepted"

    rules = propose_from_dictionary(columns, dictionary)
    origin = {}
    for r in rules:
        origin[r.id] = "concept dictionary"
    if llm_model:
        draft = SchemaProfile(profile_id="draft", columns=columns)
        try:
            known = {(r.expr, r.when) for r in rules}
            for r in propose_from_llm(gateway, llm_model, draft):
                if (r.expr, r.when) not in known:
                    rules.append(r)
                    origin[r.id] = llm_label
        except GatewayError as exc:
            notes.append(f"The local LLM could not propose rules: {exc}")

    renamed = []
    for i, r in enumerate(rules, start=1):
        origin[f"R{i}"] = origin.pop(r.id)
        renamed.append(replace(r, id=f"R{i}"))

    return {
        "columns": columns,
        "by_column": by_column,
        "combined": combined,
        "status": status,
        "rules": renamed,
        "rule_status": {r.id: "suggested" for r in renamed},
        "rule_origin": origin,
        "changes": [],
        "methods": methods,
        "llm_label": llm_label,
        "notes": notes,
    }


# --- Page setup -----------------------------------------------------------------------

ui.page_header(
    "Review columns and rules",
    ":material/fact_check:",
    "These are suggestions from the rules method, the embedding classifier and, when it is "
    "running, a local LLM. "
    "Correct anything that's wrong, then approve. Only the approved profile is used to "
    "generate data.",
)

# The step bar opens this page only once a schema is uploaded (see main.py)
schema = st.session_state["schema"]
dictionary = load_dictionary()
models, threshold = load_settings()
gateway = load_gateway(models["ollama_url"])

pulled = gateway.available_models()
llm_options = {"Off": None}
for size, tag in models["models"].items():
    if tag in pulled:
        llm_options[f"{size.capitalize()} ({tag})"] = tag

with st.expander("**Inference settings**", expanded=False, icon=":material/tune:"):
    choice = st.selectbox(
        "Local LLM used for column meanings and rule suggestions",
        list(llm_options),
        index=1 if len(llm_options) > 1 else 0,
        help="Only models already pulled into the local Ollama server are listed.",
    )
    if len(llm_options) == 1:
        st.caption(
            "No local LLM found. Start Ollama and pull a model listed in config/models.yaml "
            "to add LLM suggestions."
        )
    rerun = st.button("Run inference again", icon=":material/refresh:")

fp = fingerprint(schema)
review = st.session_state.get("review")
if review is None or review["fingerprint"] != fp or rerun:
    llm_model = llm_options[choice]
    llm_label = f"local LLM ({choice.split(' ')[0].lower()})" if llm_model else ""
    embedding, _ = load_embedding(dictionary, models["embeddings"])
    with st.spinner("Identifying columns and proposing rules..."):
        review = run_inference(
            schema, dictionary, gateway, embedding, llm_model, llm_label, threshold
        )
    review["fingerprint"] = fp
    review["profile_id"] = profile_id_from_filename(st.session_state.get("schema_name", "profile"))
    st.session_state["review"] = review

columns = review["columns"]
status = review["status"]
rules = review["rules"]
rule_status = review["rule_status"]
changes = review["changes"]

version = next_version(PROFILE_STORE, review["profile_id"])
used = ", ".join(METHOD_LABELS[m] for m in review["methods"])
n_review = sum(1 for s in status.values() if s == "review")
n_rules = sum(1 for s in rule_status.values() if s != "rejected")
n_approved = sum(1 for s in rule_status.values() if s == "approved")
st.markdown(
    f":blue-badge[:material/description: Profile {review['profile_id']}] "
    f":gray-badge[Version {version} · draft] "
    f":gray-badge[:material/psychology: Methods: {used}] "
    f":gray-badge[:material/check_circle: {len(columns) - n_review} of {len(columns)} columns "
    f"accepted] :gray-badge[:material/rule: {n_approved} of {n_rules} rules approved]"
)
embedding_problem = load_embedding(dictionary, models["embeddings"])[1]
if embedding_problem:
    st.caption(f":material/info: The embedding classifier is not used. {embedding_problem}")
for note in review["notes"]:
    st.warning(note, icon=":material/warning:")


# --- Callbacks --------------------------------------------------------------------------


def add_dictionary_rules() -> None:
    """After a meaning changes, suggest dictionary rules that now apply."""
    known = {(r.expr, r.when) for r in rules}
    for r in propose_from_dictionary(columns, dictionary):
        if (r.expr, r.when) in known:
            continue
        new_id = f"R{len(rules) + 1}"
        rules.append(replace(r, id=new_id))
        rule_status[new_id] = "suggested"
        review["rule_origin"][new_id] = "concept dictionary"


def on_meaning_change(index: int, key: str) -> None:
    old = columns[index]
    new_concept = st.session_state[key]
    if new_concept == old.concept:
        return
    columns[index] = build_column(
        old.name,
        new_concept,
        dictionary,
        schema.sample if schema.has_rows else None,
        old.inferred_by,
    )
    status[old.name] = "accepted"
    changes.append(
        {"item": f"column:{old.name}", "field": "concept", "from": old.concept, "to": new_concept}
    )
    add_dictionary_rules()


def accept_column(name: str) -> None:
    status[name] = "accepted"
    changes.append({"item": f"column:{name}", "action": "accepted as suggested"})


def set_rule_status(rule_id: str, new_status: str) -> None:
    rule_status[rule_id] = new_status
    changes.append({"item": f"rule:{rule_id}", "action": new_status})


# --- Columns ----------------------------------------------------------------------------


def type_text(col) -> str:
    text = col.type.capitalize()
    if col.type == "decimal" and col.decimals is not None:
        text += f" ({col.decimals} dp)"
    return text + (", optional" if col.nullable else "")


def format_text(col) -> str:
    seen = sorted(schema.sample[col.name].dropna().unique()) if schema.has_rows else []
    entry = dictionary.get(col.concept) if col.concept else None
    if col.pattern:
        return f"`{col.pattern}`"
    if col.allowed_values:
        return "From dictionary list"
    if col.type == "date":
        fmt = entry.default_format if entry and entry.default_format else "%Y-%m-%d"
        return fmt.replace("%Y", "YYYY").replace("%m", "MM").replace("%d", "DD")
    if col.type == "category" and seen:
        shown = ", ".join(seen[:3]) + (", ..." if len(seen) > 3 else "")
        return f"{shown} ({len(seen)} seen)"
    if col.type in {"integer", "decimal"} and seen:
        nums = schema.sample[col.name].dropna().astype(float)
        return f"{nums.min():,g} to {nums.max():,g} seen"
    return "Any"


def review_reason(name: str) -> str:
    result = review["combined"][name]
    if result.concept is None:
        col = next(c for c in columns if c.name == name)
        return (
            f"No method matched `{name}` to a known concept, so it is treated as generic "
            f"{col.type}. Pick a meaning, or accept it as it is."
        )
    others = [
        f'the {METHOD_LABELS[inf.method]} suggested "{concept_label(inf.concept)}"'
        for inf in review["by_column"][name]
        if inf.concept != result.concept
    ]
    if others:
        return f"For `{name}`, " + " and ".join(others) + ". Pick the right meaning to accept it."
    return (
        f"Confidence for `{name}` is {result.confidence:.2f}, below {threshold:.2f}. "
        "Check the meaning, then accept it."
    )


options = [None, *sorted((c.concept for c in dictionary), key=concept_label)]


def rules_using(name: str) -> list[RuleSpec]:
    """Active rules whose expression or condition mentions this column."""
    return [
        r
        for r in rules
        if rule_status[r.id] != "rejected" and name in columns_in(r.expr) | columns_in(r.when)
    ]


def example_values(name: str, limit: int = 5) -> str:
    """The column's first example values as inline code, or why there are none."""
    if not schema.has_rows:
        return "No example rows were uploaded."
    values = schema.sample[name].head(limit)
    shown = ["(empty)" if pd.isna(v) or str(v) == "" else str(v) for v in values]
    return " ".join("`" + v.replace("`", "'") + "`" for v in shown)


list_pane, detail_pane = st.columns([1, 2.2])

# Left pane: every column, with its status; picking one opens it on the right
with list_pane, ui.card("columns", height="stretch"):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Columns", icon=":material/view_column:", anchor=False)
        st.space("stretch")
        st.caption(f"{len(columns) - n_review} of {len(columns)} accepted", width="content")
    show = st.segmented_control(
        "Show columns",
        ["All", "Needs review", "Accepted"],
        default="All",
        required=True,
        key=f"filter_{fp}",
        label_visibility="collapsed",
    )
    names = [
        col.name
        for col in columns
        if show == "All" or (show == "Needs review") == (status[col.name] == "review")
    ]
    if not names:
        st.caption(f'No columns to show under "{show}".')
        picked = None
    else:
        pick_key = f"pick_{fp}"
        if st.session_state.get(pick_key) not in names:
            # Open the first column that needs review, else the first one shown
            first = next((n for n in names if status[n] == "review"), names[0])
            st.session_state[pick_key] = first
        by_name = {c.name: c for c in columns}
        picked = st.radio(
            "Column",
            names,
            key=pick_key,
            format_func=lambda n: (
                f"**`{n}`** :green[:material/check_circle:]"
                if status[n] == "accepted"
                else f"**`{n}`** :orange[:material/flag:]"
            ),
            captions=[concept_label(by_name[n].concept) for n in names],
            label_visibility="collapsed",
            width="stretch",
        )

# Right pane: the picked column's meaning, how sure each method was, examples and rules
with detail_pane, ui.card("column_detail", height="stretch"):
    if picked is None:
        st.caption("Pick a column on the left to see its details.")
    else:
        i = next(k for k, c in enumerate(columns) if c.name == picked)
        col = columns[i]
        result = review["combined"][col.name]
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"`{col.name}`", anchor=False, width="content")
            if status[col.name] == "accepted":
                st.badge("Accepted", icon=":material/check:", color="green")
            else:
                st.badge("Needs review", icon=":material/flag:", color="orange")

        if status[col.name] == "review":
            st.warning(review_reason(col.name), icon=":material/flag:")

        meaning_col, type_col, format_col = st.columns(3, vertical_alignment="top")
        key = f"meaning_{fp}_{i}"
        meaning_col.selectbox(
            "Meaning",
            options,
            index=options.index(col.concept) if col.concept in options else 0,
            format_func=concept_label,
            key=key,
            on_change=on_meaning_change,
            args=(i, key),
            help="Changing the meaning accepts the column and suggests the dictionary's rules.",
        )
        with type_col:
            st.caption("**Type**")
            st.markdown(type_text(col))
        with format_col:
            st.caption("**Format or range**")
            st.markdown(format_text(col))

        st.markdown("**How sure each method was**")
        for inf in review["by_column"][col.name]:
            agrees = inf.concept == result.concept
            st.progress(
                max(0.0, min(1.0, float(inf.confidence))),
                text=(
                    f"{METHOD_LABELS[inf.method].capitalize()}: "
                    f"{concept_label(inf.concept)} · {inf.confidence:.2f}"
                    + ("" if agrees else " · :orange[disagrees]")
                ),
            )
        if "llm" not in review["methods"]:
            st.caption(
                ":material/info: Local LLM: not used (Ollama isn't running, or it is "
                "switched off in Inference settings)."
            )
        st.caption(
            f"Combined: {result.confidence:.2f} · {result.methods_agree} of "
            f"{result.methods_total} methods agree"
        )

        st.markdown("**Example values**")
        st.markdown(example_values(col.name))

        st.markdown("**Rules that use this column**")
        used_by = rules_using(col.name)
        if not used_by:
            st.caption("No rule uses this column yet.")
        for r in used_by:
            label = "Approved" if rule_status[r.id] == "approved" else "Suggested"
            colour = "green" if label == "Approved" else "orange"
            when = f", when {code(r.when)}" if r.when else ""
            st.markdown(
                f"**{r.id}** · {r.description} · {code(r.expr)}{when} :{colour}-badge[{label}]"
            )

        with st.container(horizontal=True):
            if status[col.name] == "review":
                st.button(
                    "Accept column",
                    key=f"accept_{fp}_{i}",
                    type="primary",
                    icon=":material/check:",
                    help=f"Accept the suggested meaning for {col.name}",
                    on_click=accept_column,
                    args=(col.name,),
                )
            add_rule_here = st.button(
                "Add a rule", key=f"add_rule_{fp}_{i}", icon=":material/add:"
            )


# --- Rules ------------------------------------------------------------------------------


@st.dialog("Rule")
def rule_dialog(rule_id: str | None) -> None:
    existing = next((r for r in rules if r.id == rule_id), None)
    with st.form("rule_form", border=False):
        types = list(RULE_TYPE_LABELS)
        rule_type = st.selectbox(
            "Rule type",
            types,
            index=types.index(existing.type) if existing else 0,
            format_func=RULE_TYPE_LABELS.get,
        )
        description = st.text_input("Description", existing.description if existing else "")
        expr = st.text_input(
            "Must hold",
            existing.expr if existing else "",
            help="For example: hearing_date > issue_date",
        )
        when = st.text_input(
            "Only when (optional)",
            (existing.when or "") if existing else "",
            help="For example: status == 'Listed'",
        )
        st.caption(
            "Use column names, numbers, quoted text, null, == != < <= > >=, is null, "
            "is not null, in [...], and, or, not, unique(column). Put a column name with "
            "spaces in backticks, for example `` `Case Reference` ``."
        )
        saved = st.form_submit_button("Save rule", type="primary", icon=":material/save:")
    if not saved:
        return
    if not description.strip() or not expr.strip():
        st.error("Enter a description and an expression.", icon=":material/error:")
        return
    new_id = rule_id or f"R{len(rules) + 1}"
    rule = RuleSpec(
        id=new_id,
        type=rule_type,
        description=description.strip(),
        expr=expr.strip(),
        when=when.strip() or None,
        source=existing.source if existing else "user",
    )
    problem = rule_error(rule, [c.name for c in columns])
    if problem:
        st.error(problem, title="This rule can't be checked", icon=":material/error:")
        return
    if existing:
        rules[rules.index(existing)] = rule
        changes.append(
            {
                "item": f"rule:{new_id}",
                "action": "edited",
                "from": {"expr": existing.expr, "when": existing.when},
                "to": {"expr": rule.expr, "when": rule.when},
            }
        )
    else:
        rules.append(rule)
        rule_status[new_id] = "approved"
        review["rule_origin"][new_id] = "you"
        changes.append(
            {"item": f"rule:{new_id}", "action": "added", "expr": rule.expr, "when": rule.when}
        )
    st.rerun()


typed_sample = coerce_sample(schema.sample, columns) if schema.has_rows else None


def evidence_text(rule: RuleSpec) -> tuple[str, bool]:
    if typed_sample is None:
        return "No example rows to check", False
    try:
        evidence = check_against_sample(rule, typed_sample)
    except (KeyError, TypeError, ValueError):
        return "Could not check against the example rows", False
    return evidence.summary, evidence.fails


if picked is not None and add_rule_here:
    rule_dialog(None)

with ui.card("rules"):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Cross-column rules", icon=":material/rule:", anchor=False)
        st.space("stretch")
        if st.button("Add a rule", icon=":material/add:"):
            rule_dialog(None)

    active = [r for r in rules if rule_status[r.id] != "rejected"]
    if not active:
        st.caption("No rules proposed. Add any rules the data must follow.")
    for r in active:
        state = rule_status[r.id]
        text, fails = evidence_text(r)
        source = f"Source: {review['rule_origin'][r.id]} · {text}"
        with st.container(border=True):
            left, right = st.columns([4, 1.4], vertical_alignment="center")
            with left:
                st.markdown(f"**{r.id} · {r.description}**")
                when = f", when {code(r.when)}" if r.when else ""
                st.markdown(f"{code(r.expr)}{when}")
                st.caption(f":orange[:material/warning: {source}]" if fails else source)
            with (
                right,
                st.container(
                    horizontal=True, horizontal_alignment="right", vertical_alignment="center"
                ),
            ):
                if state == "approved":
                    st.badge("Approved", icon=":material/check:", color="green")
                    if st.button(
                        "Edit",
                        key=f"edit_{fp}_{r.id}",
                        icon=":material/edit:",
                        help=f"Edit rule {r.id}",
                    ):
                        rule_dialog(r.id)
                else:
                    st.badge("Suggested", icon=":material/lightbulb:", color="orange")
                    st.button(
                        "Approve",
                        key=f"approve_{fp}_{r.id}",
                        icon=":material/check:",
                        help=f"Approve rule {r.id}",
                        on_click=set_rule_status,
                        args=(r.id, "approved"),
                    )
                    st.button(
                        "Reject",
                        key=f"reject_{fp}_{r.id}",
                        icon=":material/close:",
                        help=f"Reject rule {r.id}",
                        on_click=set_rule_status,
                        args=(r.id, "rejected"),
                    )

    rejected = [r for r in rules if rule_status[r.id] == "rejected"]
    if rejected:
        with st.expander(f"Rejected rules ({len(rejected)})", icon=":material/block:"):
            for r in rejected:
                left, right = st.columns([4, 1.4], vertical_alignment="center")
                left.markdown(f"{r.id} · {r.description} · {code(r.expr)}")
                right.button(
                    "Restore",
                    icon=":material/undo:",
                    key=f"restore_{fp}_{r.id}",
                    help=f"Restore rule {r.id}",
                    on_click=set_rule_status,
                    args=(r.id, "suggested"),
                )


# --- Contradiction check and approval ---------------------------------------------------


def show_trial(trial: TrialResult) -> None:
    """The trial-generation check: which approved rules keep failing in trial rows."""
    if trial.error:
        st.error(trial.error, title="Trial rows could not be made", icon=":material/error:")
        return
    met = f"{trial.rows_meeting_all:,} of {trial.n_rows:,} trial rows met every approved rule."
    if not trial.problems:
        st.caption(f":material/science: Trial run: {met}")
        return
    lines = []
    for f in trial.problems:
        if not f.used_in_generation:
            why = "the generator can't build rows for it, so they are only checked"
        elif f.related:
            why = f"it likely contradicts {', '.join(f.related)} or a column's settings"
        else:
            why = "it likely contradicts a column's settings"
        lines.append(
            f"- **{f.rule_id}** ({f.description}): failed in {f.rows_failed:,} of "
            f"{f.rows_covered:,} rows it applies to; {why}."
        )
    st.warning(
        "\n".join(lines) + f"\n\n{met} Check these rules before approving. "
        "Rows that break a rule are rejected during generation, so a rule like this "
        "makes runs slow or incomplete.",
        title="Some rules failed in trial rows",
        icon=":material/science:",
    )


approved_rules = [replace(r, approved=True) for r in rules if rule_status[r.id] == "approved"]
draft = SchemaProfile(profile_id=review["profile_id"], columns=columns, rules=approved_rules)
conflicts = find_simple_conflicts(draft)
trial = run_trial(yaml.safe_dump(draft.to_dict())) if approved_rules and not conflicts else None
n_pending = sum(1 for s in rule_status.values() if s == "suggested")

blockers = []
if n_review:
    blockers.append(f"{n_review} column{'s' if n_review > 1 else ''}")
if n_pending:
    blockers.append(f"{n_pending} suggested rule{'s' if n_pending > 1 else ''}")

with ui.card("approve"):
    st.subheader("Approve profile", icon=":material/verified:", anchor=False)
    if conflicts:
        st.error(
            "\n".join(f"- {c}" for c in conflicts),
            title="Approved rules conflict",
            icon=":material/error:",
        )
    elif blockers:
        st.warning(
            f"Resolve {' and '.join(blockers)} before approving.",
            title="No conflicts between approved rules",
            icon=":material/pending_actions:",
        )
    else:
        st.success(
            "Every column and rule is resolved.",
            title="No conflicts between approved rules",
            icon=":material/check_circle:",
        )
    if trial is not None:
        show_trial(trial)

    left, right, action = st.columns([2, 2, 1], vertical_alignment="bottom")
    profile_id = left.text_input(
        "Profile name",
        review["profile_id"],
        help="Approved profiles are saved as new versions under data/profiles/.",
    )
    approver = right.text_input("Approved by", placeholder="Your name or tester ID")
    ready = not blockers and not conflicts and approver.strip() and profile_id.strip()
    approve = action.button(
        "Approve profile",
        type="primary",
        icon=":material/arrow_forward:",
        icon_position="right",
        disabled=not ready,
        width="stretch",
    )
    if approve:
        profile = replace(
            draft,
            profile_id=profile_id_from_filename(profile_id),
            version=next_version(PROFILE_STORE, profile_id_from_filename(profile_id)),
            approved_by=approver.strip(),
            approved_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        path = save_profile(profile, changes, PROFILE_STORE)
        st.session_state["profile"] = profile
        st.toast(
            f"Approved {profile.profile_id} version {profile.version}", icon=":material/verified:"
        )

if "profile" in st.session_state:
    approved = st.session_state["profile"]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.success(
            f"Saved to data/profiles/{approved.profile_id}/v{approved.version}.yaml.",
            title=f"Approved {approved.profile_id} version {approved.version}",
            icon=":material/verified:",
        )
        if st.button(
            "Generate data", type="primary", icon=":material/arrow_forward:", icon_position="right"
        ):
            st.switch_page(ui.GENERATE)
