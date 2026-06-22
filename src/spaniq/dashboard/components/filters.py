"""Shared sidebar filter components used across all dashboard pages."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from spaniq.dashboard.config import DashboardConfig

METRICS = [
    "ResponseDriftMetric",
    "SemanticSimilarityMetric",
    "OutputStabilityMetric",
    "ConsistencyMetric",
]


@dataclass
class FilterState:
    component: str | None
    metric: str
    last_n: int
    auto_refresh: bool


def render_filters(config: DashboardConfig, show_metric: bool = True) -> FilterState:
    """Render sidebar filters and return the selected state."""
    try:
        from spaniq.monitor.timeline_store import TimelineStore
        store = TimelineStore(config.db_path)
        components = store.components()
    except Exception:
        components = []

    component_options = ["All"] + components
    selected_component_label = st.sidebar.selectbox("Component", component_options)
    selected_component = None if selected_component_label == "All" else selected_component_label

    selected_metric = METRICS[0]
    if show_metric:
        selected_metric = st.sidebar.selectbox("Metric", METRICS)

    last_n = st.sidebar.slider("Last N traces", min_value=20, max_value=1000,
                                value=200, step=20)

    auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
    if auto_refresh:
        st.sidebar.caption(f"Refreshing every {config.refresh_interval}s")

    return FilterState(
        component=selected_component,
        metric=selected_metric,
        last_n=last_n,
        auto_refresh=auto_refresh,
    )
