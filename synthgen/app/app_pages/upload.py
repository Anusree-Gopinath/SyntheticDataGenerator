"""Step 1: upload and screening. Uploaded rows live in ``st.session_state`` only."""

import hashlib

import streamlit as st

from synthgen.app import ui
from synthgen.ingest import load_csv, screen_values
from synthgen.ingest.loader import MAX_COLUMNS, MAX_ROWS

HOW_IT_WORKS = [
    (":material/upload_file:", "Upload", "a fictional schema, with optional made-up rows"),
    (":material/fact_check:", "Review", "the column meanings and rules the system infers"),
    (":material/play_circle:", "Generate", "data with several approaches under the same rules"),
    (":material/monitoring:", "Compare", "realism, rule compliance, privacy risk and run time"),
]

ui.page_header(
    "Upload a fictional schema",
    ":material/upload_file:",
    "Upload a CSV with your column headers. You can add a few made-up example rows to help "
    "the system learn formats.",
)

upload_col, guide_col = st.columns([3, 2])
with upload_col:
    st.warning(
        "Example rows stay in this session's memory only. They are never saved to disk and "
        "never sent to any external service.",
        title="Never upload real case data",
        icon=":material/warning:",
    )
    with ui.card("upload_file"):
        uploaded = st.file_uploader(
            "**Schema file (CSV)**",
            type=["csv"],
            help=f"Up to {MAX_COLUMNS} columns and {MAX_ROWS} example rows.",
        )

with guide_col, ui.card("upload_guide", height="stretch"):
    st.subheader("How it works", icon=":material/route:", anchor=False)
    for number, (icon, verb, rest) in enumerate(HOW_IT_WORKS, start=1):
        st.markdown(f"{icon} **{number}. {verb}** {rest}")
    st.caption(
        "People approve, AI proposes: nothing is generated until you approve the profile. "
        "Every language model runs on this machine."
    )

if uploaded is not None:
    try:
        schema = load_csv(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()

    findings = screen_values(schema.sample) if schema.has_rows else []
    # A file with flagged values reaches the Review screen only after the user confirms
    # the values are fictional. Until then, any earlier schema is cleared so the step bar
    # keeps Review locked.
    file_id = hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]
    confirmed = st.session_state.get(f"fictional_{file_id}", False)
    if findings and not confirmed:
        st.session_state.pop("schema", None)
        st.session_state.pop("schema_name", None)
    else:
        st.session_state["schema"] = schema
        st.session_state["schema_name"] = uploaded.name

    with st.container(horizontal=True):
        st.metric("Columns", len(schema.headers), border=True)
        st.metric("Example rows", len(schema.sample), border=True)
        st.metric(
            "Values that look like personal data",
            len(findings),
            border=True,
            help="Names, addresses, phone numbers, emails and National Insurance numbers.",
        )

    with ui.card("upload_screen"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Preview", icon=":material/table_view:", anchor=False)
            st.space("stretch")
            st.caption(f"`{uploaded.name}`")
        st.dataframe(
            schema.sample if schema.has_rows else {"columns": schema.headers},
            hide_index=True,
            alt=f"Preview of {uploaded.name}",
        )

    with ui.card("upload_preview"):
        if findings:
            st.error(
                "Remove these values, or confirm the data is fictional, before continuing.",
                title=f"{len(findings)} values look like real personal data",
                icon=":material/gpp_bad:",
            )
            st.dataframe(
                [f.__dict__ for f in findings],
                hide_index=True,
                alt="Values flagged by the personal data check",
            )
            with st.container(horizontal=True, vertical_alignment="center"):
                st.checkbox(
                    "I confirm every flagged value is made up, not real personal data",
                    key=f"fictional_{file_id}",
                    help="If any value is real, remove it from the file and upload it again.",
                )
                st.space("stretch")
                if st.button(
                    "Identify columns",
                    type="primary",
                    icon=":material/arrow_forward:",
                    icon_position="right",
                    disabled=not confirmed,
                    key="identify_flagged",
                ):
                    st.switch_page(ui.REVIEW)
        else:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.success(
                    "No values look like real names, addresses, phone numbers, emails or "
                    "National Insurance numbers.",
                    title="Personal data check passed",
                    icon=":material/verified_user:",
                )
                if st.button(
                    "Identify columns",
                    type="primary",
                    icon=":material/arrow_forward:",
                    icon_position="right",
                ):
                    st.switch_page(ui.REVIEW)
