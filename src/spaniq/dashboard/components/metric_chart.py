"""Reusable Plotly drift chart for a single metric series."""

from __future__ import annotations

import plotly.graph_objects as go


def build_drift_chart(
    timestamps: list[str],
    scores: list[float],
    metric: str,
    component: str,
    threshold: float | None = None,
    alarm_indices: list[int] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(scores))),
            y=scores,
            mode="lines+markers",
            name=metric,
            line={"color": "#7c3aed", "width": 2},
            marker={"size": 4},
            hovertemplate="%{y:.4f}<extra></extra>",
        )
    )

    if threshold is not None:
        fig.add_hline(
            y=threshold * 0.5, line_dash="dot", line_color="orange", annotation_text="mild drift"
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="threshold")

    if alarm_indices:
        for idx in alarm_indices:
            if 0 <= idx < len(scores):
                fig.add_vline(x=idx, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=f"{metric} — {component}",
        xaxis_title="Trace index",
        yaxis_title="Score",
        template="plotly_dark",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        showlegend=False,
    )
    return fig
