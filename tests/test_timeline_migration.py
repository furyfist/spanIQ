"""Tests for TimelineStore component column migration."""
import pytest
from spaniq.monitor.timeline_store import TimelineStore


@pytest.fixture
def store(tmp_path):
    return TimelineStore(str(tmp_path / "test.db"))


def test_fresh_db_has_component_column(store):
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(timeline)")}
    assert "component" in cols


def test_record_default_component(store):
    store.record(
        trace_id="t1", baseline_id="b1", metric_name="TestMetric",
        score=0.8, threshold=0.5, passed=True, timestamp="2024-01-01T00:00:00+00:00"
    )
    rows = store.query("TestMetric")
    assert len(rows) == 1
    assert rows[0].component == "default"


def test_record_named_component(store):
    store.record(
        trace_id="t1", baseline_id="b1", metric_name="TestMetric",
        score=0.9, threshold=0.5, passed=True, timestamp="2024-01-01T00:00:00+00:00",
        component="retrieval",
    )
    rows = store.query("TestMetric", component="retrieval")
    assert len(rows) == 1
    assert rows[0].component == "retrieval"


def test_v2_query_unaffected(store):
    store.record(
        trace_id="t1", baseline_id="b1", metric_name="M",
        score=0.5, threshold=0.5, passed=True, timestamp="2024-01-01T00:00:00+00:00"
    )
    store.record(
        trace_id="t2", baseline_id="b1", metric_name="M",
        score=0.6, threshold=0.5, passed=True, timestamp="2024-01-02T00:00:00+00:00",
        component="retrieval",
    )
    rows = store.query("M")
    assert len(rows) == 2


def test_query_series_by_component(store):
    for i in range(5):
        store.record(
            trace_id=f"t{i}", baseline_id="b1", metric_name="M",
            score=float(i), threshold=0.5, passed=True,
            timestamp=f"2024-01-0{i+1}T00:00:00+00:00",
            component="retrieval",
        )
    series = store.query_series("retrieval", "M", last_n=10)
    assert len(series) == 5
    assert series[0] == 0.0
    assert series[-1] == 4.0
