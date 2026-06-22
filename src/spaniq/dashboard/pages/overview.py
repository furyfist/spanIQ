"""Overview page — system health summary."""
from __future__ import annotations

import streamlit as st

from spaniq.dashboard.config import DashboardConfig
from spaniq.dashboard.components.health_badge import render_health_badge

METRICS = [
    "ResponseDriftMetric",
    "SemanticSimilarityMetric",
    "OutputStabilityMetric",
]


def render(config: DashboardConfig) -> None:
    st.title("📊 spanIQ — Overview")

    try:
        from spaniq.monitor.timeline_store import TimelineStore
        store = TimelineStore(config.db_path)
    except Exception as exc:
        st.error(f"Could not open database `{config.db_path}`: {exc}")
        return

    total = store.count()
    components = store.components()
    alert_count = _count_alerts(config.db_path)

    # Top-level KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total traces", f"{total:,}")
    col2.metric("Components", len(components))
    col3.metric("Active alerts", alert_count, delta=None)

    st.markdown("---")

    if not components:
        st.info("No data yet. Run `spaniq monitor run` or `spaniq collect-otel` to ingest traces.")
        return

    # Per-component health table
    st.subheader("Component health")
    for comp in components:
        with st.expander(f"**{comp}**", expanded=True):
            cols = st.columns(len(METRICS))
            for i, metric in enumerate(METRICS):
                rows = store.query(metric, last_n=1, component=comp)
                with cols[i]:
                    if rows:
                        r = rows[-1]
                        render_health_badge(comp, metric, r.score, r.threshold, r.passed)
                    else:
                        st.caption(f"{metric[:20]}: no data")

    # Ingestion sparkline (last hour approximate)
    st.markdown("---")
    st.subheader("Recent ingestion rate")
    _render_ingestion_chart(store)

    if config.refresh_interval > 0:
        import time
        st.caption(f"Auto-refreshing every {config.refresh_interval}s")
        time.sleep(config.refresh_interval)
        st.rerun()


def _count_alerts(db_path: str) -> int:
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE severity='warning' OR severity='error'"
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _render_ingestion_chart(store) -> None:
    import plotly.graph_objects as go

    rows = store.query("ResponseDriftMetric", last_n=100)
    if not rows:
        st.caption("No traces yet.")
        return

    timestamps = [r.timestamp[:16] for r in rows]
    fig = go.Figure(go.Scatter(
        x=list(range(len(timestamps))),
        y=[1] * len(timestamps),
        mode="markers",
        marker={"color": "#7c3aed", "size": 6},
        hovertext=timestamps,
        hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=120,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        showlegend=False,
        yaxis={"visible": False},
        xaxis_title="Recent traces",
    )
    st.plotly_chart(fig, use_container_width=True)
