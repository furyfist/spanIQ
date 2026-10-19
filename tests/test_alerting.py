from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from spaniq.monitor.alerting import AlertEngine


def _make_mr(name: str, score: float, threshold: float, passed: bool):
    mr = MagicMock()
    mr.metric_name = name
    mr.score = score
    mr.threshold = threshold
    mr.passed = passed
    return mr


def _check(engine, passed: bool, name="ResponseDriftMetric", ts="2024-01-01T00:00:00+00:00"):
    mr = _make_mr(name, score=0.2 if not passed else 0.01, threshold=0.10, passed=passed)
    engine.check([mr], trace_id="t1", timestamp=ts)


def test_alert_fires_after_n_consecutive(tmp_path):
    engine = AlertEngine(alert_after=3, alerts_path=str(tmp_path / "alerts.jsonl"))
    for _ in range(3):
        _check(engine, passed=False)
    assert len(engine.alerts) == 1


def test_no_alert_on_intermittent_failure(tmp_path):
    engine = AlertEngine(alert_after=3, alerts_path=str(tmp_path / "alerts.jsonl"))
    _check(engine, passed=False)
    _check(engine, passed=True)
    _check(engine, passed=False)
    assert len(engine.alerts) == 0


def test_alert_contains_correct_fields(tmp_path):
    engine = AlertEngine(alert_after=1, alerts_path=str(tmp_path / "alerts.jsonl"))
    _check(engine, passed=False, ts="2024-06-01T12:00:00+00:00")
    alert = engine.alerts[0]
    assert alert.metric_name == "ResponseDriftMetric"
    assert alert.score == pytest.approx(0.2)
    assert alert.threshold == pytest.approx(0.10)
    assert alert.trace_id == "t1"
    assert alert.timestamp == "2024-06-01T12:00:00+00:00"


def test_multiple_metrics_independent(tmp_path):
    engine = AlertEngine(alert_after=3, alerts_path=str(tmp_path / "alerts.jsonl"))
    # fail metric A 3 times → 1 alert
    for _ in range(3):
        _check(engine, passed=False, name="MetricA")
    # fail metric B only 2 times → no alert for B
    for _ in range(2):
        _check(engine, passed=False, name="MetricB")
    assert len(engine.alerts) == 1
    assert engine.alerts[0].metric_name == "MetricA"


def test_alert_appended_to_jsonl(tmp_path):
    path = str(tmp_path / "alerts.jsonl")
    engine = AlertEngine(alert_after=1, alerts_path=path)
    _check(engine, passed=False)
    with open(path) as f:
        line = json.loads(f.read().strip())
    assert line["metric_name"] == "ResponseDriftMetric"
    assert "ALERT" in line["message"]


def test_counter_resets_on_pass(tmp_path):
    engine = AlertEngine(alert_after=3, alerts_path=str(tmp_path / "alerts.jsonl"))
    _check(engine, passed=False)
    _check(engine, passed=False)
    _check(engine, passed=True)  # reset
    _check(engine, passed=False)
    _check(engine, passed=False)
    assert len(engine.alerts) == 0  # never reached 3 in a row after reset


def test_one_alert_per_incident(tmp_path):
    """A long failure streak is a single incident, not an alert per trace."""
    engine = AlertEngine(alert_after=3, alerts_path=str(tmp_path / "alerts.jsonl"))
    for _ in range(20):
        _check(engine, passed=False)
    assert len(engine.alerts) == 1
    _check(engine, passed=True)
    for _ in range(3):
        _check(engine, passed=False)
    assert len(engine.alerts) == 2


def test_repeat_every_realerts_during_long_incident(tmp_path):
    engine = AlertEngine(alert_after=3, alerts_path=str(tmp_path / "a.jsonl"), repeat_every=5)
    for _ in range(13):
        _check(engine, passed=False)
    assert [a.consecutive_count for a in engine.alerts] == [3, 8, 13]


def test_alert_written_to_sqlite(tmp_path):
    from spaniq.monitor.timeline_store import TimelineStore

    db = str(tmp_path / "spaniq.db")
    engine = AlertEngine(alert_after=1, alerts_path=str(tmp_path / "a.jsonl"), db_path=db)
    _check(engine, passed=False)
    assert TimelineStore(db).alert_count() == 1
