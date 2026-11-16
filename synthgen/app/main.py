"""SynthGen review app: navigation across the seven core steps.

Run: ``make app`` (or ``streamlit run synthgen/app/main.py``).
Each screen is in ``app_pages/``. Uploaded rows live in ``st.session_state`` only.
Steps are reached through the step bar under the header: only Upload is open at first,
and each later step opens once the one before it is done.
"""

import streamlit as st

from synthgen.app import ui

st.set_page_config(page_title="SynthGen", page_icon=":material/dataset:", layout="wide")

pages = [st.Page(s.path, title=s.label, icon=s.icon, help=s.help) for s in ui.STEPS]
page = st.navigation(pages, position="hidden")

# A locked step (for example opened by its URL) goes back to the furthest open step
index = next(i for i, p in enumerate(pages) if p.url_path == page.url_path)
is_open = ui.steps_open()
if not is_open[index]:
    st.switch_page(ui.STEPS[max(i for i, ok in enumerate(is_open) if ok)].path)

ui.styles()
ui.app_header()
st.space("small")  # a little more room between the header and the step bar
ui.step_bar(ui.STEPS[index].path)

page.run()
