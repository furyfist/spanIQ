"""Tests for dashboard overview data logic (Step 10) — no Streamlit import."""

from __future__ import annotations

import pytest

from spaniq.monitor.timeline_store import TimelineStore


@pytest.fixture()
def populated_db(tmp_path):
    db = str(tmp_path / "test.db")
    store = TimelineStore(db)
    for i in range(10):
        store.record(
            trace_id=f"t{i}",
            baseline_id="b1",
            metric_name="ResponseDriftMetric",
            score=0.05 + i * 0.01,
            threshold=0.2,
            passed=True,
            timestamp=f"2024-01-01T00:00:{i:02d}",
            component="retriever",
        )
    return db, store


def test_total_trace_count(populated_db):
    _, store = populated_db
    assert store.count() == 10


def test_components_returns_correct_list(populated_db):
    _, store = populated_db
    comps = store.components()
    assert "retriever" in comps


def test_health_badge_color_green():
    from spaniq.dashboard.components.health_badge import health_color

    assert health_color(0.05, 0.2) == "green"


def test_alert_count_empty_db(populated_db):
    db, store = populated_db
    assert store.alert_count() == 0
