"""VentureIQ — application entry point.  Run:  python -m streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="VentureIQ", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

from ui import theme  # noqa: E402
from ui.page_explore import page as explore_page  # noqa: E402
from ui.page_misc import PAGES, history, overview, settings  # noqa: E402
from ui.page_predict import page as predict_page  # noqa: E402
from ui import state  # noqa: E402

ROUTES = {"Overview": overview, "Predict Startup": predict_page, "Explore Startup Idea": explore_page,
          "Analysis History": history, "Settings & Status": settings}


def main() -> None:
    theme.inject(st)
    theme.intro_once(st)
    st.session_state.setdefault("nav", PAGES[0])
    with st.sidebar:
        st.markdown(theme.flat('<div class="viq-brand"><div class="viq-word">Venture<span>IQ</span></div><small>Startup evaluation &amp; risk intelligence</small></div>'),
                    unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        live = state.get_llm() is not None
        st.caption(f"AI generation: **{'live' if live else 'offline (no API key)'}**")
        st.caption("Estimates are decision-support aids, not advice.")
    try:
        ROUTES[st.session_state["nav"]]()
    except Exception:  # noqa: BLE001 - never show a stack trace or secrets to the user
        import logging
        logging.getLogger("ventureiq").exception("Unhandled error while rendering page")
        st.error("Something went wrong while rendering this page. Details were written to the terminal log. "
                 "Check Settings & Status for missing files, then try again.")


main()
