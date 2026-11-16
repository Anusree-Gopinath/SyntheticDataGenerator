"""Shared look for the review app: page paths, the header, the step bar and cards."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

UPLOAD = "app_pages/upload.py"
REVIEW = "app_pages/review.py"
GENERATE = "app_pages/generate.py"
REPORT = "app_pages/report.py"


@dataclass(frozen=True)
class Step:
    path: str
    label: str
    icon: str
    help: str


STEPS = [
    Step(
        UPLOAD,
        "1 · Upload",
        ":material/upload_file:",
        "Upload a fictional CSV schema and check it for anything that looks like real "
        "personal data.",
    ),
    Step(
        REVIEW,
        "2–4 · Review and approve",
        ":material/fact_check:",
        "Check the column meanings and rules the system inferred, correct them, and approve "
        "the profile.",
    ),
    Step(
        GENERATE,
        "5–6 · Generate",
        ":material/play_circle:",
        "Pick a scenario and approaches, then generate rows that all pass the approved rules.",
    ),
    Step(
        REPORT,
        "7 · Quality report",
        ":material/monitoring:",
        "Compare approaches on rule compliance, realism, privacy risk and run time.",
    ),
]


def steps_done() -> list[bool]:
    """Which steps this session has finished: upload, approve, generate (report never ends)."""
    state = st.session_state
    runs = state.get("runs") or {}
    return ["schema" in state, "profile" in state, bool(runs.get("records")), False]


def steps_open() -> list[bool]:
    """Upload is always open; every other step opens once the one before it is done."""
    done = steps_done()
    return [True, *done[:3]]


def styles() -> None:
    """Card and step-bar looks from the mockup that the theme can't set.

    The page background, font and badge colours come from ``.streamlit/config.toml``. Only
    keyed elements are styled (``.st-key-<key>``), so Streamlit's own classes are never
    targeted. In the dark choice (a grey theme with black text) cards are a lighter grey.
    """
    dark = st.context.theme.type == "dark"
    card = "#E2E6EB" if dark else "#FFFFFF"
    done = "#166534" if dark else "#15803D"
    edge = "#9AA3AF" if dark else "#D1D5DB"  # step bar border: light grey
    st.html(
        f"""<style>
        .st-key-app_header, [class*="st-key-card_"], .st-key-step_bar {{
            background: {card};
            border-radius: 12px;
        }}
        .st-key-step_bar {{
            border: 1px solid {edge};
            box-shadow: 0 0 10px 2px rgba(107, 114, 128, 0.18);
        }}
        .st-key-step_current a, [class*="st-key-step_open_"] a {{
            border-radius: 999px;
            padding: 2px 12px;
        }}
        .st-key-step_current a {{ background: #2563EB; }}
        .st-key-step_current a, .st-key-step_current a * {{ color: #FFFFFF !important; }}
        [class*="st-key-step_open_"] a {{ background: #DBEAFE; }}
        [class*="st-key-step_open_"] a, [class*="st-key-step_open_"] a * {{
            color: #1E3A8A !important;
        }}
        [class*="st-key-step_done_"] a, [class*="st-key-step_done_"] a * {{
            color: {done} !important;
        }}
        </style>"""
    )


def card(key: str, **kwargs):
    """A bordered section drawn as a white card. ``key`` must be unique on the page."""
    return st.container(border=True, key=f"card_{key}", **kwargs)


def app_header() -> None:
    """The band at the top of every screen: app name, what it does, and the local-only badge."""
    with st.container(
        border=True, horizontal=True, vertical_alignment="center", key="app_header"
    ):
        st.title("SynthGen", icon=":material/dataset:", anchor=False, width="content")
        st.caption(
            "LLM-guided synthetic test data for Automated Software Testing",
            width="content",
        )
        st.space("stretch")
        st.badge(
            "Local only: no data leaves this machine",
            icon=":material/lock:",
            color="green",
            help="All language models run locally. Nothing is sent to an external service.",
        )


def step_bar(current: str) -> None:
    """Bold step tabs. A step stays locked until the one before it is done; finished steps
    get a tick and the current step is highlighted. The bar sits in a card with a light
    grey border and a faint grey glow (``step_bar`` in ``styles``)."""
    done, is_open = steps_done(), steps_open()
    with st.container(
        border=True, horizontal=True, vertical_alignment="center", gap="xsmall", key="step_bar"
    ):
        for i, step in enumerate(STEPS):
            if i:
                st.markdown(":gray[:material/arrow_forward:]", width="content")
            if not is_open[i]:
                icon = ":material/lock:"
                hint = "Finish the step before this one first."
            elif done[i]:
                icon = ":material/check_circle:"
                hint = step.help
            else:
                icon = step.icon
                hint = step.help
            if step.path == current:
                key = "step_current"
            elif not is_open[i]:
                key = f"step_locked_{i}"
            elif done[i]:
                key = f"step_done_{i}"
            else:
                key = f"step_open_{i}"
            with st.container(key=key, width="content"):
                st.page_link(
                    step.path,
                    label=f"**{step.label}**",
                    icon=icon,
                    help=hint,
                    disabled=not is_open[i],
                )


def page_header(title: str, icon: str, subtitle: str) -> None:
    """Screen title, then a one-line subtitle."""
    st.header(title, icon=icon, anchor=False)
    st.caption(subtitle)
