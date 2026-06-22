"""Alert Log page — sortable/filterable table of fired alerts."""
from __future__ import annotations

import io

import streamlit as st

from spaniq.dashboard.config import DashboardConfig
from spaniq.dashboard.components.filters import METRICS


def render(config: DashboardConfig) -> None:
    st.title("🔔 Alert Log")

    try:
        from spaniq.monitor.timeline_store import TimelineStore
        store = TimelineStore(config.db_path)
    except Exception as exc:
        st.error(f"Could not open database: {exc}")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        components = ["All"] + store.components()
        sel_comp = st.selectbox("Component", components)
    with col2:
        sel_metric = st.selectbox("Metric", ["All"] + METRICS)
    with col3:
        last_n = st.slider("Max rows", 20, 500, 100, 20)

    alerts = store.query_alerts(
        component=None if sel_comp == "All" else sel_comp,
        metric_name=None if sel_metric == "All" else sel_metric,
        last_n=last_n,
    )

    if not alerts:
        st.success("No alerts recorded yet. All systems healthy.")
        return

    st.metric("Alerts shown", len(alerts))

    import pandas as pd
    df = pd.DataFrame(alerts)
    display_cols = ["timestamp", "metric_name", "component", "score",
                    "threshold", "severity", "consecutive_count", "message"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols]
    df["timestamp"] = df["timestamp"].str[:19]

    st.dataframe(
        df.style.apply(_color_severity, axis=1, subset=["severity"]),
        use_container_width=True,
        height=400,
    )

    # CSV export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export CSV",
        data=csv,
        file_name="spaniq_alerts.csv",
        mime="text/csv",
    )


def _color_severity(row):
    color = {"warning": "background-color: #78350f", "error": "background-color: #7f1d1d"}.get(
        str(row.iloc[0]), ""
    )
    return [color]
