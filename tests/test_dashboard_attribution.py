"""Tests for dashboard attribution page data logic (Step 12)."""

from __future__ import annotations

import pytest

from spaniq.monitor.timeline_store import TimelineStore


@pytest.fixture()
def two_component_db(tmp_path):
    db = str(tmp_path / "attr.db")
    store = TimelineStore(db)
    for i in range(60):
        # retriever drifts after trace 30
        score = 0.05 if i < 30 else 0.25
        store.record(
            trace_id=f"t{i}",
            baseline_id="b1",
            metric_name="ResponseDriftMetric",
            score=score,
            threshold=0.2,
            passed=(score <= 0.2),
            timestamp=f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}",
            component="retriever",
        )
        store.record(
            trace_id=f"t{i}",
            baseline_id="b1",
            metric_name="ResponseDriftMetric",
            score=0.05,
            threshold=0.2,
            passed=True,
            timestamp=f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}",
            component="generator",
        )
    return db, store


def test_attribution_runs_with_two_components(two_component_db):
    db, store = two_component_db
    from spaniq.attribution.attributor import attribute

    result = attribute(
        timeline=store,
        components=store.components(),
        metrics=["ResponseDriftMetric"],
        last_n=60,
        warmup=5,
    )
    assert result is not None
    assert result.verdict is not None and len(result.verdict) > 0


def test_no_degradation_case(tmp_path):
    db = str(tmp_path / "clean.db")
    store = TimelineStore(db)
    for i in range(40):
        store.record(
            trace_id=f"t{i}",
            baseline_id="b1",
            metric_name="ResponseDriftMetric",
            score=0.05,
            threshold=0.2,
            passed=True,
            timestamp=f"2024-01-01T00:00:{i:02d}",
            component="llm",
        )
    from spaniq.attribution.attributor import attribute

    result = attribute(
        timeline=store,
        components=["llm"],
        metrics=["ResponseDriftMetric"],
        last_n=40,
        warmup=5,
    )
    assert "no degradation" in result.verdict.lower()


def test_single_component_pipeline(tmp_path):
    db = str(tmp_path / "single.db")
    store = TimelineStore(db)
    for i in range(30):
        store.record(
            trace_id=f"t{i}",
            baseline_id="b1",
            metric_name="ResponseDriftMetric",
            score=0.05 + (0.3 if i > 15 else 0.0),
            threshold=0.2,
            passed=(i <= 15),
            timestamp=f"2024-01-01T00:00:{i:02d}",
            component="only",
        )
    from spaniq.attribution.attributor import attribute

    result = attribute(
        timeline=store,
        components=["only"],
        metrics=["ResponseDriftMetric"],
        last_n=30,
        warmup=5,
    )
    assert result is not None


def test_confidence_score_in_result(two_component_db):
    _, store = two_component_db
    from spaniq.attribution.attributor import attribute

    result = attribute(
        timeline=store,
        components=store.components(),
        metrics=["ResponseDriftMetric"],
        last_n=60,
        warmup=5,
    )
    # confidence lives on root_cause ComponentBreak, not the top-level result
    if result.root_cause is not None:
        assert 0.0 <= result.root_cause.confidence <= 1.0
    else:
        assert result.verdict is not None
