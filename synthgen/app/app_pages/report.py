"""Step 7: comparative quality report.

Compares the approaches in one batch of runs: rule compliance, rejections, realism against
held-back statistics and the reference dataset, usefulness of patterns and run time, each as
the mean and 95% confidence interval across seeds. Privacy results appear for Scenario 2.
This page is an evaluator, so it may read the evaluation store; generators never do.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from synthgen.app import ui
from synthgen.core.models import SchemaProfile
from synthgen.core.profile_manager import coerce_sample
from synthgen.evaluation.evaluate import (
    IDENTIFIERS,
    EvaluationStores,
    evaluate_run,
    reference_baseline,
    reference_types,
)
from synthgen.evaluation.report import build_report, compare_approaches, write_report
from synthgen.orchestration.approaches import APPROACHES
from synthgen.reference_world.stores import profile_column_map

ROOT = Path(__file__).resolve().parents[3]
RUN_STORE = ROOT / "data" / "runs"
PROFILE_STORE = ROOT / "data" / "profiles"
DATA = ROOT / "data"
LABELS = {a.key: a.label for a in APPROACHES}
# Single-series bar colour (categorical slot 1), with its own step for dark mode
BAR = {"light": "#2a78d6", "dark": "#3987e5"}
INK = {"light": ("#0b0b0b", "#52514e", "#e6e5e1"), "dark": ("#f0efec", "#c3c2b7", "#2e2e2c")}


def approach_label(name: str) -> str:
    if name.startswith("llm:"):
        return f"LLM · {name[4:]}"
    return LABELS.get(name, name.replace("_", " ").capitalize())


@st.cache_data(show_spinner=False)
def load_stores(scenario: int) -> EvaluationStores:
    thresholds = yaml.safe_load((ROOT / "config" / "thresholds.yaml").read_text())
    stores = EvaluationStores(
        reference=pd.read_csv(DATA / "reference" / "reference.csv", dtype=str),
        held_back=yaml.safe_load((DATA / "evaluation" / "held_back_stats.yaml").read_text()),
        privacy_percentile=float(thresholds["privacy_distance"]["percentile"]),
    )
    if scenario == 2:
        stores.sample = pd.read_csv(DATA / "calibration" / "scenario2_sample.csv", dtype=str)
        stores.candidates = pd.read_csv(DATA / "evaluation" / "candidate_pool.csv", dtype=str)
    return stores


@st.cache_data(show_spinner=False)
def baseline(profile_path: str) -> tuple[str | None, pd.DataFrame | None, float]:
    return reference_baseline(load_stores(1), SchemaProfile.load(profile_path))


@st.cache_data(show_spinner=False)
def evaluate_file(csv_path: str, mtime: float, profile_path: str, scenario: int) -> dict:
    """Metrics for one run; cached until the file changes (``mtime``)."""
    profile = SchemaProfile.load(profile_path)
    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    typed = coerce_sample(data, profile.columns)
    _, test, _ = baseline(profile_path)
    return evaluate_run(typed, profile, load_stores(scenario), test)


def batches() -> list[dict]:
    found = []
    for path in sorted(RUN_STORE.glob("*/batch.yaml"), reverse=True):
        batch = yaml.safe_load(path.read_text())
        batch["folder"] = str(path.parent)  # the folder may have moved since the run
        found.append(batch)
    return found


def batch_label(b: dict) -> str:
    approaches = dict.fromkeys(approach_label(r["approach"]) for r in b["records"])
    names = ", ".join(approaches)
    return f"{b['batch']} · {b['profile']} · {names or 'no finished runs'}"


def fmt(row: pd.Series, kind: str) -> str:
    if pd.isna(row["mean"]):
        return "–"
    mean, half = row["mean"], (row["hi"] - row["lo"]) / 2
    if kind == "pct":
        text, err = f"{mean:.1%}", f"{half * 100:.1f} pts"
    elif kind == "int":
        text, err = f"{mean:,.0f}", f"{half:,.0f}"
    elif kind == "sec":
        text, err = f"{mean:.2f}", f"{half:.2f}"
    else:
        text, err = f"{mean:.3f}", f"{half:.3f}"
    return text if row["n"] < 2 else f"{text} ± {err}"


def wide(summary: pd.DataFrame, columns: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """One row per approach; ``columns`` maps metric -> (heading, number kind)."""
    rows = []
    for approach, part in summary.groupby("approach", sort=False):
        by_metric = part.set_index("metric")
        row = {"Approach": approach}
        for metric, (heading, kind) in columns.items():
            row[heading] = fmt(by_metric.loc[metric], kind) if metric in by_metric.index else "–"
        rows.append(row)
    return pd.DataFrame(rows)


LABELS_TESTED = {
    "rows_rejected": "Rows rejected",
    "held_back_gap": "Gap from held-back statistics",
    "reference_distance": "Distance from reference",
    "tstr_accuracy": "Train on synthetic, test on reference",
    "seconds": "Run time (s)",
    "near_copy_share": "Near copies",
    "distance_attack_auc": "Distance attack AUC",
    "density_attack_auc": "Density attack AUC",
}


def significance_table(runs: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame | None:
    """Readable paired-test results, or None when there is nothing to compare."""
    if runs["approach"].nunique() < 2 or runs.groupby("approach").size().max() < 2:
        return None
    tests = compare_approaches(runs, list(labels))
    if tests.empty:
        return None

    def yes_no(value) -> str:
        return "–" if value is None or pd.isna(value) else ("Yes" if value else "No")

    return pd.DataFrame(
        {
            "Metric": tests["metric"].map(labels),
            "Comparison": tests["approach_a"] + " − " + tests["approach_b"],
            "Difference": [
                "–" if pd.isna(d) else f"{d:.3f} ({lo:.3f} to {hi:.3f})"
                for d, lo, hi in zip(tests["difference"], tests["lo"], tests["hi"], strict=True)
            ],
            "Seeds paired": tests["pairs"],
            "p (Holm)": tests["p_holm"].map(lambda p: "–" if pd.isna(p) else f"{p:.3f}"),
            "Significant": tests["significant"].map(yes_no),
        }
    )


def ci_bar_chart(data: pd.DataFrame, category: str, value_name: str, x_title: str) -> alt.Chart:
    """Horizontal bars of ``mean`` per ``category`` with 95% CI whiskers and value labels."""
    theme = getattr(getattr(st.context, "theme", None), "type", None) or "light"
    text, muted, grid = INK.get(theme, INK["light"])
    data = data.dropna(subset=["mean"]).copy()
    data["label"] = data["mean"].map(lambda v: f"{v:.3f}")
    data["ci"] = data.apply(lambda r: f"{r['lo']:.3f} to {r['hi']:.3f}", axis=1)
    y = alt.Y(f"{category}:N", sort=None, title=None, axis=alt.Axis(labelColor=text, ticks=False))
    tooltip = [
        alt.Tooltip(f"{category}:N", title=category.capitalize()),
        alt.Tooltip("label:N", title=value_name),
        alt.Tooltip("ci:N", title="95% CI"),
        alt.Tooltip("n:Q", title="Seeds"),
    ]
    x = alt.X(
        "mean:Q",
        title=x_title,
        scale=alt.Scale(domainMin=0),
        axis=alt.Axis(labelColor=muted, titleColor=muted, gridColor=grid, domain=False),
    )
    bars = (
        alt.Chart(data)
        .mark_bar(size=20, cornerRadiusEnd=4, color=BAR.get(theme, BAR["light"]))
        .encode(x=x, y=y, tooltip=tooltip)
    )
    whiskers = (
        alt.Chart(data)
        .mark_rule(color=muted, strokeWidth=2)
        .encode(x="lo:Q", x2="hi:Q", y=y, tooltip=tooltip)
    )
    labels = (
        alt.Chart(data)
        .mark_text(align="left", dx=6, color=text)
        .encode(x="hi:Q", y=y, text="label:N")
    )
    return (bars + whiskers + labels).properties(height=44 * len(data) + 40)


def render(scenario: int, all_batches: list[dict]) -> None:
    options = [b for b in all_batches if b["scenario"] == scenario and b["records"]]
    if not options:
        st.info(
            "Generate data for this scenario to compare approaches here.",
            title=f"No finished runs for Scenario {scenario} yet",
            icon=":material/info:",
        )
        if st.button("Generate data", icon=":material/play_circle:", key=f"go_{scenario}"):
            st.switch_page(ui.GENERATE)
        return
    current = (st.session_state.get("runs") or {}).get("batch")
    labels = [batch_label(b) for b in options]
    index = next((i for i, b in enumerate(options) if b["batch"] == current), 0)
    batch_col, _ = st.columns([3, 2])
    picked = batch_col.selectbox(
        "Batch of runs",
        labels,
        index=index,
        key=f"batch_{scenario}",
        help="Each batch is one Start runs on the generate screen, saved under data/runs/.",
    )
    chosen = options[labels.index(picked)]

    folder = Path(chosen["folder"])
    profile_id, version = chosen["profile"].rsplit("@", 1)
    profile_path = PROFILE_STORE / profile_id / f"v{version}.yaml"
    if not profile_path.exists():  # batches from the experiment runner keep their own copy
        profile_path = folder / "profile.yaml"
    if not profile_path.exists():
        st.error(
            "It should be under data/profiles/.",
            title=f"Profile {chosen['profile']} is missing",
            icon=":material/error:",
        )
        return
    profile = SchemaProfile.load(profile_path)

    approaches = {r["approach"] for r in chosen["records"]}
    with st.container(horizontal=True):
        st.metric("Approaches", len(approaches), border=True)
        st.metric("Seeds per approach", chosen["n_seeds"], border=True)
        st.metric("Rows per run", f"{chosen['n_rows']:,}", border=True)
        st.metric(
            "Rows rejected and regenerated",
            f"{sum(r['rows_rejected'] for r in chosen['records']):,}",
            border=True,
            help="Across all runs in this batch.",
        )

    rows, incomplete = [], []
    with st.spinner("Evaluating runs against the evaluation store..."):
        for record in chosen["records"]:
            csv = folder / f"{record['run_id']}.csv"
            if not csv.exists():
                st.warning(
                    f"Data for {record['run_id']} is missing; it is left out.",
                    icon=":material/warning:",
                )
                continue
            metrics = evaluate_file(str(csv), csv.stat().st_mtime, str(profile_path), scenario)
            label = approach_label(record["approach"])
            rows.append(
                {
                    "approach": label,
                    "seed": record["seed"],
                    "rows_rejected": record["rows_rejected"],
                    "seconds": record["duration_seconds"],
                }
                | metrics
            )
            if not record["complete"]:
                incomplete.append(f"{label} seed {record['seed']}")
    if not rows:
        return
    runs = pd.DataFrame(rows)
    summary = build_report(runs)
    target, _, reference_accuracy = baseline(str(profile_path))

    table = wide(
        summary,
        {
            "rule_compliance": ("Rule compliance", "pct"),
            "rows_rejected": ("Rows rejected", "int"),
            "held_back_gap": ("Gap from held-back statistics", "num"),
            "reference_distance": ("Distance from reference", "num"),
            "tstr_accuracy": ("Train on synthetic, test on reference", "pct"),
            "seconds": ("Run time (s)", "sec"),
        },
    )
    with ui.card(f"rep_summary_{scenario}"):
        st.subheader("Summary by approach", icon=":material/table_chart:", anchor=False)
        st.dataframe(table, hide_index=True, width="stretch", alt="Summary of metrics by approach")
        checks = int(runs["held_back_checks"].max())
        notes = [
            (
                "Rule compliance should be 100% for every approach because failing rows are "
                "regenerated; the rejection count shows how well each approach learned the rules."
            ),
            (
                f"Gap from held-back statistics averages the {checks} held-back statistic"
                f"{'s' if checks != 1 else ''} this schema's columns allow (0 = matches)."
                if checks
                else "None of the held-back statistics can be computed from this schema's "
                "columns, so there is no gap to show."
            ),
            (
                "Distance from reference is the mean KS (numbers, dates) or Jensen-Shannon "
                "(categories) distance per column. Lower distances are better."
            ),
        ]
        if target:
            notes.append(
                f"Train on synthetic: a simple model predicts `{target}` and is tested on "
                f"half the reference world. Trained on the other half instead, it scores "
                f"{reference_accuracy:.1%}."
            )
        st.caption("± values are 95% confidence intervals across seeds.")
        with st.expander("How to read these figures", icon=":material/help:"):
            for note in notes:
                st.markdown(f"- {note}")
        significance = significance_table(runs, LABELS_TESTED)
        if significance is not None:
            with st.expander("Are the differences significant?", icon=":material/functions:"):
                st.dataframe(
                    significance,
                    hide_index=True,
                    width="stretch",
                    alt="Paired t-tests between approaches across seeds",
                )
                st.caption(
                    "Paired t-tests: runs with the same seed form a pair. P-values are "
                    f"Holm-adjusted for the {len(significance['Comparison'].unique())} "
                    "comparisons per metric. With few seeds, only large differences can reach "
                    "significance, so 'No' means not shown, not the same. A dash means no test "
                    "was possible (fewer than two pairs, or the same difference every time)."
                )
        if incomplete:
            st.warning(
                "Stopped short of the requested rows after the retry limit: "
                + ", ".join(incomplete)
                + ". Their rules may be too strict for the approach, or contradict each other.",
                title="Some runs are incomplete",
                icon=":material/warning:",
            )

    left, right = st.columns(2)
    with left, ui.card(f"rep_realism_{scenario}", height="stretch"):
        st.subheader("Gap from held-back statistics", icon=":material/bar_chart:", anchor=False)
        st.caption("Mean across seeds, with 95% CI · shorter is better")
        if summary[summary["metric"] == "held_back_gap"]["mean"].notna().any():
            st.altair_chart(
                ci_bar_chart(
                    summary[summary["metric"] == "held_back_gap"],
                    "approach",
                    "Mean gap",
                    "Mean gap (0 = matches held-back statistics)",
                ),
                width="stretch",
                alt="Bar chart of the mean gap from held-back statistics for each approach",
            )
        else:
            st.caption("No held-back statistic can be computed from this schema's columns.")

    privacy_table = None
    with right, ui.card(f"rep_privacy_{scenario}", height="stretch"):
        st.subheader("Privacy risk", icon=":material/shield:", anchor=False)
        if scenario == 1:
            st.info(
                icon=":material/info:",
                body="Privacy tests run in Scenario 2 only, where sample rows influence generation. "
                "Switch to the Scenario 2 tab to see membership attack AUC, distance to "
                "closest record and exact copies.",
            )
        else:
            privacy_table = wide(
                summary,
                {
                    "exact_copies": ("Exact copies", "int"),
                    "near_copy_share": ("Near copies", "pct"),
                    "dcr_median": ("Median distance to closest record", "num"),
                    "distance_attack_auc": ("Distance attack AUC", "num"),
                    "density_attack_auc": ("Density attack AUC", "num"),
                },
            )
            st.dataframe(
                privacy_table,
                hide_index=True,
                width="stretch",
                alt="Privacy test results by approach",
            )
            threshold = runs["privacy_threshold"].iloc[0]
            compared = [c for c in reference_types(profile) if c not in IDENTIFIERS]
            st.caption(
                f"Rows are compared on {len(compared)} column"
                f"{'s' if len(compared) != 1 else ''} the reference world shares with this "
                f"schema ({', '.join(f'`{c}`' for c in compared)}); with few or mostly empty "
                "columns, generated rows can match sample rows by chance. Near copies are "
                f"rows closer than {threshold:.3f} (Gower distance, the 5th percentile "
                "between different sample rows) to a sample row. An AUC near 0.5 means the "
                "attacker does little better than guessing."
            )
        st.caption(
            ":material/gavel: Results are evidence about membership inference and record copying under the "
            "stated threat model and test conditions. They are not proof that the data is "
            "private, and other attacks such as attribute inference are not tested."
        )

    # Realism by column
    distances = summary[summary["metric"].str.startswith("distance:")].copy()
    if not distances.empty:
        to_profile = {ref: col for ref, col in profile_column_map(profile).items()}
        distances["column"] = (
            distances["metric"]
            .str.removeprefix("distance:")
            .map(lambda ref: to_profile.get(ref, ref))
        )
        with ui.card(f"rep_detail_{scenario}"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader("Realism by column", icon=":material/view_column:", anchor=False)
                st.space("stretch")
                names = list(dict.fromkeys(distances["approach"]))
                shown = names[0]
                if len(names) > 1:
                    shown = st.segmented_control(
                        "Approach",
                        names,
                        default=names[0],
                        key=f"cols_{scenario}",
                        label_visibility="collapsed",
                        required=True,
                    )
            st.caption(
                "Distance from the reference world for each column the two share: KS for "
                "numbers and dates, Jensen-Shannon for categories. Mean across seeds, with 95% "
                "CI · shorter is better. Long bars show where the approach misses reality."
            )
            per_column = distances[distances["approach"] == shown].sort_values(
                "mean", ascending=False
            )
            st.altair_chart(
                ci_bar_chart(
                    per_column, "column", "Distance", "Distance from reference (0 = same)"
                ),
                width="stretch",
                alt=f"Bar chart of each column's distance from the reference world for {shown}",
            )

    # Run record and downloads
    tables = {"summary": table} | ({"privacy": privacy_table} if privacy_table is not None else {})
    if significance is not None:
        tables["significance"] = significance
    html = write_report(tables, folder, f"SynthGen quality report: {chosen['batch']}")
    models = sorted({r["llm_model"] for r in chosen["records"] if r.get("llm_model")})
    code = sorted({r["code_version"] for r in chosen["records"] if r.get("code_version")})
    with ui.card(f"rep_download_{scenario}"):
        st.subheader("Run record and downloads", icon=":material/download:", anchor=False)
        st.caption(
            f"Run record: profile {chosen['profile']} · code version "
            f"{', '.join(code) or 'not recorded'} · models {', '.join(models) or 'none'} · "
            f"saved in {folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder}"
        )
        run_ids = [r["run_id"] for r in chosen["records"]]
        with st.container(horizontal=True, vertical_alignment="bottom"):
            run_id = st.selectbox("Run to download", run_ids, key=f"run_{scenario}", width=420)
            csv = folder / f"{run_id}.csv"
            st.download_button(
                "Download data (CSV)",
                csv.read_bytes() if csv.exists() else b"",
                file_name=csv.name,
                mime="text/csv",
                icon=":material/table_view:",
                disabled=not csv.exists(),
                key=f"data_{scenario}",
            )
            st.space("stretch")
            st.download_button(
                "Download report",
                html.read_bytes(),
                file_name=f"report-{chosen['batch']}.html",
                mime="text/html",
                type="primary",
                icon=":material/download:",
                key=f"report_{scenario}",
            )
        with st.expander("Run records", icon=":material/receipt_long:"):
            st.json(chosen["records"], expanded=False)


ui.page_header(
    "Comparative quality report",
    ":material/monitoring:",
    "Each approach compared on rule compliance, realism against held-back statistics and the "
    "reference world, usefulness of patterns, privacy risk and run time.",
)

if not (DATA / "evaluation" / "held_back_stats.yaml").exists():
    st.error(
        "Build it with `make reference`.",
        title="The evaluation store is missing",
        icon=":material/error:",
    )
    st.stop()

found = batches()
tab_labels = [":material/looks_one: Scenario 1: no rows", ":material/looks_two: Scenario 2: sample"]
for scenario, tab in zip((1, 2), st.tabs(tab_labels), strict=True):
    with tab:
        render(scenario, found)
