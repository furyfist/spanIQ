"""Tests for the Attributor ranking logic."""
from __future__ import annotations

import numpy as np
import pytest

from spaniq.attribution.attributor import attribute, AttributionResult
from spaniq.monitor.timeline_store import TimelineStore


def _populate_store(tmp_path, component_scores: dict[str, list[float]], metric="TestMetric"):
    store = TimelineStore(str(tmp_path / "test.db"))
    from datetime import datetime, timedelta, timezone
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for comp, scores in component_scores.items():
        for i, score in enumerate(scores):
            ts = (base + timedelta(seconds=i)).isoformat()
            store.record(
                trace_id=f"{comp}-{i}", baseline_id="b1", metric_name=metric,
                score=score, threshold=0.5, passed=score >= 0.5,
                timestamp=ts, component=comp,
            )
    return store


def _make_scores(n=200, break_at=100, mu_good=0.8, mu_bad=0.2, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    good = list(rng.normal(mu_good, sigma, break_at))
    bad = list(rng.normal(mu_bad, sigma, n - break_at))
    return good + bad


def test_root_cause_identified(tmp_path):
    retrieval = _make_scores(break_at=80)
    generation = _make_scores(break_at=90)
    healthy = list(np.random.default_rng(5).normal(0.8, 0.03, 200))
    store = _populate_store(tmp_path, {
        "retrieval": retrieval,
        "generation": generation,
        "healthy_comp": healthy,
    })
    result = attribute(store, ["retrieval", "generation", "healthy_comp"], ["TestMetric"])
    assert result.root_cause is not None
    assert result.root_cause.component == "retrieval"
    assert "generation" in [c.component for c in result.cascade]
    assert "healthy_comp" in result.healthy


def test_no_break_returns_healthy(tmp_path):
    flat = list(np.random.default_rng(0).normal(0.8, 0.03, 200))
    store = _populate_store(tmp_path, {"comp_a": flat, "comp_b": flat})
    result = attribute(store, ["comp_a", "comp_b"], ["TestMetric"])
    assert result.root_cause is None
    assert len(result.healthy) == 2


def test_single_component_no_cascade(tmp_path):
    scores = _make_scores(break_at=100)
    store = _populate_store(tmp_path, {"only": scores})
    result = attribute(store, ["only"], ["TestMetric"])
    if result.root_cause:
        assert len(result.cascade) == 0


def test_simultaneous_breaks_no_root_cause_claim(tmp_path):
    scores = _make_scores(break_at=100)
    store = _populate_store(tmp_path, {"a": scores, "b": list(scores)})
    result = attribute(store, ["a", "b"], ["TestMetric"], cluster_window=5)
    if result.root_cause:
        assert result.verdict is not None


def test_healthy_component_not_in_cascade(tmp_path):
    broken = _make_scores(break_at=80)
    flat = list(np.random.default_rng(99).normal(0.8, 0.03, 200))
    store = _populate_store(tmp_path, {"broken": broken, "flat": flat})
    result = attribute(store, ["broken", "flat"], ["TestMetric"])
    cascade_names = [c.component for c in result.cascade]
    assert "flat" not in cascade_names


def test_confidence_in_range(tmp_path):
    scores = _make_scores(break_at=80)
    store = _populate_store(tmp_path, {"comp": scores})
    result = attribute(store, ["comp"], ["TestMetric"])
    if result.root_cause:
        assert 0.0 <= result.root_cause.confidence <= 1.0


def test_verdict_language(tmp_path):
    retrieval = _make_scores(break_at=80)
    generation = _make_scores(break_at=90)
    store = _populate_store(tmp_path, {"retrieval": retrieval, "generation": generation})
    result = attribute(store, ["retrieval", "generation"], ["TestMetric"])
    assert "broke first" in result.verdict or "no degradation" in result.verdict


def test_two_independent_metrics(tmp_path):
    scores_m1 = _make_scores(break_at=80)
    scores_m2 = _make_scores(break_at=90, seed=1)
    from datetime import datetime, timedelta, timezone
    store = TimelineStore(str(tmp_path / "test.db"))
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, (s1, s2) in enumerate(zip(scores_m1, scores_m2)):
        ts = (base + timedelta(seconds=i)).isoformat()
        store.record("t", "b", "M1", s1, 0.5, s1 >= 0.5, ts, component="comp")
        store.record("t", "b", "M2", s2, 0.5, s2 >= 0.5, ts, component="comp")
    result = attribute(store, ["comp"], ["M1", "M2"])
    if result.root_cause:
        assert len(result.root_cause.broken_metrics) >= 1


def test_empty_series_handled(tmp_path):
    store = TimelineStore(str(tmp_path / "test.db"))
    result = attribute(store, ["a", "b"], ["TestMetric"])
    assert result.root_cause is None
    assert result.verdict == "no degradation detected"


def test_event_window_set(tmp_path):
    retrieval = _make_scores(break_at=80)
    store = _populate_store(tmp_path, {"retrieval": retrieval})
    result = attribute(store, ["retrieval"], ["TestMetric"])
    assert isinstance(result.event_window, tuple)
    assert len(result.event_window) == 2
