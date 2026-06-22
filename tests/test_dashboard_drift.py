"""Tests for dashboard drift timeline chart logic (Step 11)."""
from __future__ import annotations

import pytest

from spaniq.dashboard.components.metric_chart import build_drift_chart


def _scores(n: int = 20) -> list[float]:
    return [0.05 + i * 0.005 for i in range(n)]


def test_chart_renders_with_valid_data():
    fig = build_drift_chart([], _scores(), "ResponseDriftMetric", "comp-a", threshold=0.2)
    assert fig is not None
    assert len(fig.data) >= 1


def test_chart_has_correct_series_count():
    scores = _scores(30)
    fig = build_drift_chart([], scores, "SemanticSimilarityMetric", "comp-b")
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 30


def test_threshold_lines_added_when_provided():
    fig = build_drift_chart([], _scores(), "ResponseDriftMetric", "c", threshold=0.2)
    # hlines are added as layout shapes
    shapes = fig.layout.shapes
    assert shapes is not None and len(shapes) >= 1


def test_single_point_series_no_crash():
    fig = build_drift_chart([], [0.1], "OutputStabilityMetric", "single", threshold=0.3)
    assert fig is not None
