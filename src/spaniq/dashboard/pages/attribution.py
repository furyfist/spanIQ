"""Attribution page — interactive V3 changepoint/cascade visualization."""

from __future__ import annotations

import streamlit as st

from spaniq.dashboard.config import DashboardConfig


def render(config: DashboardConfig) -> None:
    st.title("🔍 Attribution")
    st.caption(
        "Root-cause analysis powered by PELT changepoint detection + CUSUM cascade ranking (V3)."
    )

    try:
        from spaniq.monitor.timeline_store import TimelineStore

        store = TimelineStore(config.db_path)
    except Exception as exc:
        st.error(f"Could not open database: {exc}")
        return

    components = store.components()
    if not components:
        st.info("No component data. Run `spaniq pipeline run` or `spaniq collect-otel` first.")
        return

    col_l, col_r = st.columns([1, 3])
    with col_l:
        last_n = st.number_input(
            "Traces to analyse", min_value=50, max_value=2000, value=500, step=50
        )
        penalty = st.number_input("PELT penalty (0 = auto)", min_value=0.0, value=0.0, step=0.5)
        warmup = st.number_input("Warmup traces", min_value=0, max_value=100, value=20, step=5)
        run_btn = st.button("Run Attribution", type="primary")

    if not run_btn:
        st.info("Set parameters and click **Run Attribution**.")
        return

    with st.spinner("Running changepoint detection…"):
        try:
            from spaniq.attribution.attributor import attribute

            result = attribute(
                timeline=store,
                components=components,
                metrics=[
                    "ResponseDriftMetric",
                    "SemanticSimilarityMetric",
                    "OutputStabilityMetric",
                ],
                last_n=int(last_n),
                pelt_penalty=penalty if penalty > 0 else None,
                warmup=int(warmup),
            )
        except Exception as exc:
            st.error(f"Attribution failed: {exc}")
            return

    # Verdict card
    with col_r:
        _render_verdict(result)

    st.markdown("---")
    _render_cascade_chart(result)
    _render_component_table(result)


def _render_verdict(result) -> None:
    if result.verdict == "no_degradation":
        st.success("✅ No significant degradation detected across all components.")
        return

    root = next((c for c in result.components if c.role == "root_cause"), None)
    cascades = [c for c in result.components if c.role == "cascade"]

    if root:
        st.error(f"🔴 Root cause: **{root.name}** (break at trace {root.break_index})")
        if cascades:
            names = ", ".join(c.name for c in cascades)
            st.warning(f"🟡 Cascade affected: {names}")
    else:
        st.warning(f"Verdict: {result.verdict}")

    if root and hasattr(root, "confidence"):
        st.metric("Confidence", f"{root.confidence:.0%}")


def _render_cascade_chart(result) -> None:
    import plotly.graph_objects as go

    st.subheader("Component timeline")
    color_map = {"root_cause": "#ef4444", "cascade": "#f59e0b", "healthy": "#22c55e"}
    fig = go.Figure()

    for _i, comp in enumerate(result.components):
        color = color_map.get(comp.role, "#6b7280")
        break_x = comp.break_index or 0
        fig.add_trace(
            go.Bar(
                x=[break_x],
                y=[comp.name],
                orientation="h",
                marker_color=color,
                name=f"{comp.name} ({comp.role})",
                hovertemplate=f"<b>{comp.name}</b><br>Role: {comp.role}<br>Break: {break_x}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=max(200, len(result.components) * 60),
        showlegend=True,
        xaxis_title="Trace index of break",
        margin={"l": 20, "r": 20, "t": 20, "b": 40},
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_component_table(result) -> None:
    import pandas as pd

    st.subheader("Component details")
    rows = []
    for comp in result.components:
        rows.append(
            {
                "Component": comp.name,
                "Role": comp.role,
                "Break index": comp.break_index or "—",
                "Lead gap": getattr(comp, "lead_gap", "—"),
                "Metric": getattr(comp, "primary_metric", "—"),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
