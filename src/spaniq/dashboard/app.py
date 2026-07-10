"""spanIQ Streamlit dashboard — entry point.

Launch via:  spaniq dashboard  [--db spaniq.db] [--port 8501]
Or directly: streamlit run src/spaniq/dashboard/app.py -- --db spaniq.db
"""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from spaniq.dashboard.config import DashboardConfig


def _parse_args() -> DashboardConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default="spaniq.db")
    # streamlit passes its own args before "--", grab only ours
    try:
        idx = sys.argv.index("--")
        our_args = sys.argv[idx + 1 :]
    except ValueError:
        our_args = []
    args, _ = parser.parse_known_args(our_args)
    return DashboardConfig(db_path=args.db)


config = _parse_args()

st.set_page_config(
    page_title=config.page_title,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-mode accent CSS
st.markdown(
    """
    <style>
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 8px 12px; }
    .block-container { padding-top: 1.5rem; }
    .stSidebar { background: #181825; }
    div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image(
    "https://img.shields.io/badge/spanIQ-v0.4-blue?style=flat-square",
    use_column_width=False,
)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "",
    ["Overview", "Drift Timeline", "Attribution", "Alert Log"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"DB: `{config.db_path}`")

if page == "Overview":
    from spaniq.dashboard.pages.overview import render

    render(config)
elif page == "Drift Timeline":
    from spaniq.dashboard.pages.drift_timeline import render

    render(config)
elif page == "Attribution":
    from spaniq.dashboard.pages.attribution import render

    render(config)
elif page == "Alert Log":
    from spaniq.dashboard.pages.alert_log import render

    render(config)
