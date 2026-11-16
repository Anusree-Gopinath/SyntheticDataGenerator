"""Steps 5 and 6: generate and validate.

The user picks a scenario, approaches, row count and seeds. Each run goes through the
orchestrator, so every row is checked against the approved rules and failing rows are
regenerated. Generators see only the calibration store (and, in Scenario 2, the sample
from the reference world), never the evaluation store.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from synthgen.app import ui
from synthgen.core.models import SchemaProfile
from synthgen.gateway import LLMGateway
from synthgen.orchestration.approaches import APPROACHES
from synthgen.orchestration.orchestrator import run_generator
from synthgen.reference_world.stores import (
    calibration_for_profile,
    profile_column_map,
    sample_for_profile,
)

ROOT = Path(__file__).resolve().parents[3]
PROFILE_STORE = ROOT / "data" / "profiles"
CALIBRATION = ROOT / "data" / "calibration" / "calibration_stats.yaml"
SCENARIO2_SAMPLE = ROOT / "data" / "calibration" / "scenario2_sample.csv"
RUN_STORE = ROOT / "data" / "runs"
SCENARIOS = {
    1: ("Scenario 1: no rows", "Schema, approved rules and calibration statistics only"),
    2: (
        "Scenario 2: small fictional sample",
        "Also uses a 200-row sample from the reference world. Privacy tests run only here.",
    ),
}


@st.cache_resource
def load_models() -> dict:
    return yaml.safe_load((ROOT / "config" / "models.yaml").read_text())


@st.cache_resource
def load_gateway(url: str) -> LLMGateway:
    return LLMGateway(url)


def profile_label(profile_id: str, version: int) -> str:
    return f"{profile_id} · version {version}"


def saved_profiles() -> dict[str, Path]:
    """Label -> path for every profile version in the profile store, oldest first."""
    paths = [p for p in PROFILE_STORE.glob("*/v*.yaml") if p.stem[1:].isdigit()]
    paths.sort(key=lambda p: (p.parent.name, int(p.stem[1:])))
    return {profile_label(p.parent.name, int(p.stem[1:])): p for p in paths}


ui.page_header(
    "Generate synthetic data",
    ":material/play_circle:",
    "Choose a scenario and the approaches to compare. Every row is checked against the "
    "approved rules; rows that fail are regenerated.",
)

# --- Profile ---------------------------------------------------------------------------

profiles = saved_profiles()
if not profiles:
    st.info(
        "Generation uses an approved profile.",
        title="No approved profile yet",
        icon=":material/info:",
    )
    if st.button("Review columns and rules", icon=":material/fact_check:"):
        st.switch_page(ui.REVIEW)
    st.stop()

options = list(profiles)
current = st.session_state.get("profile")
default = profile_label(current.profile_id, current.version) if current else options[-1]
profile_col, _ = st.columns([2, 3])
choice = profile_col.selectbox(
    "Approved profile",
    options,
    index=options.index(default) if default in options else len(options) - 1,
    help="Profiles approved on the review screen, saved under data/profiles/.",
)
profile = SchemaProfile.load(profiles[choice])
if not profile.is_approved:
    st.error(
        "Approve it on the review screen first.",
        title="This profile is not approved",
        icon=":material/error:",
    )
    st.stop()
st.markdown(
    f":blue-badge[:material/description: Profile {profile.profile_id}] "
    f":green-badge[:material/verified: Version {profile.version} · approved by "
    f"{profile.approved_by}] "
    f":gray-badge[{len(profile.columns)} columns · {len(profile.rules)} rules]"
)

if not CALIBRATION.exists():
    st.error(
        "Build it with `make reference`.",
        title="The calibration store is missing",
        icon=":material/error:",
    )
    st.stop()

cal = yaml.safe_load(CALIBRATION.read_text())
calibration = calibration_for_profile(cal, profile)
mapped = set(profile_column_map(profile).values())
unmatched = [c.name for c in profile.columns if c.name not in mapped]
if unmatched:
    st.caption(
        f":material/info: No calibration statistics for "
        f"{', '.join(f'`{c}`' for c in unmatched)}: these columns are generated from the "
        "profile alone."
    )

# --- Setup -----------------------------------------------------------------------------

models = load_models()
pulled = load_gateway(models["ollama_url"]).available_models()
runs = st.session_state.get("runs")
running = bool(runs and runs["running"])

left, right = st.columns(2)
with left, ui.card("gen_scenario", height="stretch"):
    st.subheader("Scenario", icon=":material/science:", anchor=False)
    scenario = st.radio(
        "Scenario",
        list(SCENARIOS),
        format_func=lambda s: SCENARIOS[s][0],
        captions=[SCENARIOS[s][1] for s in SCENARIOS],
        label_visibility="collapsed",
        disabled=running,
    )
    if scenario == 2 and not SCENARIO2_SAMPLE.exists():
        st.error(
            "Build it with `make reference`.",
            title="The Scenario 2 sample is missing",
            icon=":material/error:",
        )
    rows_col, seeds_col = st.columns(2)
    n_rows = rows_col.number_input(
        "Rows per run",
        min_value=100,
        max_value=100_000,
        value=10_000,
        step=1_000,
        disabled=running,
    )
    n_seeds = seeds_col.number_input(
        "Seeds",
        min_value=1,
        max_value=20,
        value=5,
        disabled=running,
        help="Each approach runs once per seed (seeds 1, 2, ...) for confidence intervals.",
    )

with right, ui.card("gen_settings", height="stretch"):
    st.subheader("Approaches", icon=":material/compare_arrows:", anchor=False)
    chosen = []
    for a in APPROACHES:
        tag = models["models"].get(a.model_size) if a.model_size else None
        label = f"LLM · {tag} (local)" if tag else a.label
        reason = None
        if scenario not in a.scenarios:
            reason = "Scenario 2 only"
        elif tag and not pulled:
            reason = "Start the local Ollama server (`ollama serve`) first"
        elif tag and tag not in pulled:
            reason = f"Pull {tag} into the local Ollama server first (`ollama pull {tag}`)"
        ticked = st.checkbox(
            label,
            value=reason is None,
            disabled=reason is not None or running,
            key=f"approach_{a.key}_{scenario}",
            help=reason,
        )
        st.caption(a.description + (f" · :orange[{reason}]" if reason else ""))
        if ticked and reason is None:
            chosen.append(a)

# --- Run plan --------------------------------------------------------------------------

with st.container(horizontal=True):
    st.metric(
        "Runs planned",
        len(chosen) * int(n_seeds),
        border=True,
        help="One run per approach and seed.",
    )
    st.metric("Rows to generate", f"{len(chosen) * int(n_seeds) * int(n_rows):,}", border=True)
    st.metric("Approaches", len(chosen), border=True)
    st.metric(
        "Rules every row must pass",
        len(profile.rules),
        border=True,
        help="Rows that fail are dropped and regenerated, up to a retry limit.",
    )

llm_chosen = [a for a in chosen if a.model_size]
if llm_chosen:
    calls = math.ceil(int(n_rows) / 50) * int(n_seeds) * len(llm_chosen)
    st.caption(
        f":material/schedule: The LLM approaches make at least {calls:,} model calls "
        "(up to 50 rows each, fewer for wide schemas). Each call can take a minute or more on "
        "this machine, so try a few hundred rows first."
    )

# --- Runs ------------------------------------------------------------------------------


def runs_table(state: dict) -> pd.DataFrame:
    rows = []
    for s in state["approaches"].values():
        rows.append(
            {
                "Approach": s["label"],
                "Status": s["status"],
                "Progress": s["done"] / state["n_seeds"],
                "Valid rows": f"{s['valid']:,}" if s["done"] else "–",
                "Rejected and regenerated": f"{s['rejected']:,}" if s["done"] else "–",
                "Time (s)": f"{s['seconds']:.2f}" if s["done"] else "–",
            }
        )
    return pd.DataFrame(rows)


def show_table(placeholder, state: dict) -> None:
    placeholder.dataframe(
        runs_table(state),
        hide_index=True,
        width="stretch",
        alt="Progress of each approach's runs",
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Progress": st.column_config.ProgressColumn(
                "Progress", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )


# A click while runs are in progress stops the script; tidy up the interrupted state.
if running:
    for s in runs["approaches"].values():
        if s["status"] in {"Running", "Queued"}:
            s["status"] = "Stopped"
    runs["running"] = False
    running = False

with ui.card("gen_runs"):
    heading = st.empty()
    table = st.empty()
    live = st.empty()  # rows so far in the current LLM run
    errors = st.container()
    with st.container(horizontal=True, vertical_alignment="center"):
        start = st.button(
            "Start runs",
            type="primary",
            icon=":material/play_arrow:",
            disabled=not chosen or (scenario == 2 and not SCENARIO2_SAMPLE.exists()),
        )
        # Any click interrupts a script that is still running, which is what stops the runs
        st.button(
            "Stop",
            icon=":material/stop:",
            help="Stop the runs. Finished runs are kept. An LLM run stops once its current "
            "batch of rows comes back from the model.",
        )
        st.space("stretch")
        has_runs = bool(runs and any(s["done"] for s in runs["approaches"].values()))
        if has_runs and st.button("View quality report", icon=":material/monitoring:"):
            st.switch_page(ui.REPORT)

    if start:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        batch, n = f"{stamp}-s{scenario}", 1
        while (RUN_STORE / batch).exists():  # never overwrite an earlier batch
            n += 1
            batch = f"{stamp}-s{scenario}-{n}"
        folder = RUN_STORE / batch
        runs = {
            "batch": batch,
            "folder": str(folder),
            "profile": f"{profile.profile_id}@{profile.version}",
            "scenario": scenario,
            "n_rows": int(n_rows),
            "n_seeds": int(n_seeds),
            "running": True,
            "records": [],
            "approaches": {
                a.key: {
                    "label": a.label,
                    "status": "Queued",
                    "done": 0,
                    "valid": 0,
                    "rejected": 0,
                    "seconds": 0.0,
                    "incomplete": False,
                }
                for a in chosen
            },
        }
        st.session_state["runs"] = runs
        sample = (
            sample_for_profile(pd.read_csv(SCENARIO2_SAMPLE, dtype=str), profile)
            if scenario == 2
            else None
        )
        gateway = load_gateway(models["ollama_url"])
        folder.mkdir(parents=True, exist_ok=True)

        for seed in range(1, int(n_seeds) + 1):
            heading.subheader(
                f"Runs · seed {seed} of {int(n_seeds)}", icon=":material/autorenew:", anchor=False
            )
            for a in chosen:
                state = runs["approaches"][a.key]
                if state["status"] == "Failed":
                    continue
                state["status"] = "Running"
                show_table(table, runs)
                try:
                    generator = a.make(gateway, models["models"], seed)
                    if hasattr(generator, "on_batch"):  # LLM: progress, and a place to stop

                        def on_batch(have: int, want: int, label=a.label, seed=seed) -> None:
                            live.caption(
                                f":material/hourglass_top: {label}, seed {seed}: "
                                f"{have:,} of {want:,} rows from the model so far"
                            )

                        generator.on_batch = on_batch
                    data, record = run_generator(
                        generator, profile, calibration, scenario, int(n_rows), seed, sample
                    )
                    live.empty()
                # A failed approach must not stop the others (RuntimeError covers
                # NotImplementedError and GatewayError)
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    state["status"] = "Failed"
                    errors.error(
                        str(exc), title=f"{a.label}, seed {seed} failed", icon=":material/error:"
                    )
                    continue
                record.llm_model = getattr(generator, "model", None)
                data.to_csv(folder / f"{record.run_id}.csv", index=False)
                (folder / f"{record.run_id}.json").write_text(record.to_json())
                if getattr(generator, "loss_history", None):  # CTGAN or TVAE training loss per epoch
                    pd.DataFrame(generator.loss_history).to_csv(
                        folder / f"{record.run_id}.loss.csv", index=False
                    )
                runs["records"].append(asdict(record))
                state["done"] += 1
                state["valid"] = len(data)
                state["rejected"] += record.rows_rejected
                state["seconds"] += record.duration_seconds or 0.0
                if not record.complete:
                    state["incomplete"] = True
                    errors.warning(
                        icon=":material/warning:",
                        body=f"{a.label}, seed {seed}: only {len(data):,} of {int(n_rows):,} rows "
                        "passed the rules after the retry limit.",
                    )
                if seed < int(n_seeds):
                    state["status"] = "Queued"
                else:
                    state["status"] = "Incomplete" if state["incomplete"] else "Done"
                show_table(table, runs)

        manifest = {k: v for k, v in runs.items() if k != "running"}
        (folder / "batch.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        runs["running"] = False
        st.rerun()

    if runs:
        done = min(s["done"] for s in runs["approaches"].values())
        heading.subheader(
            f"Runs · {runs['n_seeds']} seeds · {runs['n_rows']:,} rows per run",
            icon=":material/list_alt:",
            anchor=False,
        )
        show_table(table, runs)
        st.caption(
            f"Scenario {runs['scenario']} · profile {runs['profile']} · saved to "
            f"{Path(runs['folder']).relative_to(ROOT)}. Rejected rows are counted across all "
            "seeds; valid rows are for the latest seed."
            + ("" if done == runs["n_seeds"] else " Some runs did not finish.")
        )
    else:
        heading.subheader("Runs", icon=":material/list_alt:", anchor=False)
        table.caption("No runs yet. Choose approaches and start the runs.")
