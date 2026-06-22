"""Health badge component — colour-coded status per component."""
from __future__ import annotations

import streamlit as st


def health_color(score: float, threshold: float) -> str:
    """Return 'green', 'yellow', or 'red' based on score relative to threshold."""
    ratio = score / threshold if threshold else 0.0
    if ratio < 0.5:
        return "green"
    if ratio < 1.0:
        return "yellow"
    return "red"


def render_health_badge(component: str, metric: str, score: float,
                         threshold: float, passed: bool) -> None:
    color = health_color(score, threshold)
    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[color]
    label = "healthy" if passed else "drifting"
    st.metric(
        label=f"{icon} {component} — {metric[:20]}",
        value=f"{score:.4f}",
        delta=f"threshold {threshold:.4f}",
        delta_color="normal" if passed else "inverse",
    )
