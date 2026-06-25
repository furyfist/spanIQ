"""Drift Timeline page — interactive metric time-series with Plotly."""
from __future__ import annotations

import streamlit as st

from spaniq.dashboard.config import DashboardConfig
from spaniq.dashboard.components.filters import render_filters
from spaniq.dashboard.components.metric_chart import build_drift_chart


def render(config: DashboardConfig) -> None:
    st.title("📈 Drift Timeline")

    filters = render_filters(config, show_metric=True)

    try:
        from spaniq.monitor.timeline_store import TimelineStore
        store = TimelineStore(config.db_path)
    except Exception as exc:
        st.error(f"Could not open database: {exc}")
        return

    components = [filters.component] if filters.component else store.components()
    if not components:
        st.info("No component data found. Ingest some traces first.")
        return

    for comp in components:
        rows = store.query(filters.metric, last_n=filters.last_n, component=comp, ascending=True)
        if not rows:
            st.caption(f"{comp}: no data for {filters.metric}")
            continue

        scores = [r.score for r in rows]
        timestamps = [r.timestamp[:19] for r in rows]
        threshold = rows[-1].threshold if rows else None

        fig = build_drift_chart(
            timestamps=timestamps,
            scores=scores,
            metric=filters.metric,
            component=comp,
            threshold=threshold,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats below each chart
        passed = sum(1 for r in rows if r.passed)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Traces", len(rows))
        c2.metric("Mean score", f"{sum(scores)/len(scores):.4f}")
        c3.metric("Pass rate", f"{passed/len(rows)*100:.1f}%")
        c4.metric("Latest", f"{scores[-1]:.4f}")
        st.markdown("---")

    if filters.auto_refresh:
        import time
        time.sleep(config.refresh_interval)
        st.rerun()
