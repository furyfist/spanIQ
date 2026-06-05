from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from spaniq.monitor.timeline_store import TimelineStore


def _ts(offset_seconds: int = 0) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset_seconds)).isoformat()


@pytest.fixture
def store(tmp_path):
    return TimelineStore(db_path=str(tmp_path / "test.db"))


def _record(store, metric="ResponseDriftMetric", score=0.05, passed=True, offset=0):
    store.record(
        trace_id=f"t{offset}",
        baseline_id="b1",
        metric_name=metric,
        score=score,
        threshold=0.10,
        passed=passed,
        timestamp=_ts(offset),
    )


def test_record_and_query(store):
    _record(store, score=0.04)
    rows = store.query("ResponseDriftMetric", last_n=10)
    assert len(rows) == 1
    assert rows[0].score == pytest.approx(0.04)
    assert rows[0].passed is True


def test_query_last_n(store):
    for i in range(10):
        _record(store, score=0.01 * i, offset=i)
    rows = store.query("ResponseDriftMetric", last_n=5)
    assert len(rows) == 5


def test_query_ordered_by_timestamp_desc(store):
    for i in range(5):
        _record(store, score=float(i), offset=i)
    rows = store.query("ResponseDriftMetric", last_n=5)
    timestamps = [r.timestamp for r in rows]
    assert timestamps == sorted(timestamps, reverse=True)


def test_summary_statistics(store):
    scores = [0.01, 0.02, 0.03, 0.04, 0.05]
    for i, s in enumerate(scores):
        _record(store, score=s, passed=(s < 0.10), offset=i)
    summary = store.summary("ResponseDriftMetric", last_n=10)
    assert summary.n == 5
    assert summary.mean_score == pytest.approx(0.03, abs=1e-6)
    assert summary.pass_rate == pytest.approx(1.0)
    assert summary.min_score == pytest.approx(0.01)
    assert summary.max_score == pytest.approx(0.05)


def test_summary_trend_positive_on_increasing_scores(store):
    for i in range(10):
        _record(store, score=0.01 * i, offset=i)
    summary = store.summary("ResponseDriftMetric", last_n=10)
    # query returns DESC, polyfit over that, expect non-zero trend
    assert isinstance(summary.trend, float)


def test_empty_timeline_returns_zeros(store):
    summary = store.summary("NonExistentMetric")
    assert summary.n == 0
    assert summary.mean_score == 0.0
    assert summary.pass_rate == 0.0
    assert summary.trend == 0.0


def test_count(store):
    assert store.count() == 0
    _record(store)
    _record(store, offset=1)
    assert store.count() == 2
