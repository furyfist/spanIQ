from __future__ import annotations

import json

import pytest

from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.file import FileCollector
from spaniq.monitor.monitor import Monitor
from spaniq.monitor.timeline_store import TimelineStore


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def baseline_name(db):
    store = BaselineStore(db)
    store.create(
        name="test-base",
        prompt="what color is the sky?",
        outputs=["the sky is blue"] * 20,
    )
    return "test-base"


def test_monitor_processes_all_traces(tmp_path, db, baseline_name):
    traces_path = tmp_path / "traces.jsonl"
    _write_jsonl(traces_path, [
        {"input": "what color is the sky?", "output": "the sky is blue"}
        for _ in range(5)
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name=baseline_name,
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run()
    assert report.total_traces == 5


def test_monitor_stores_to_timeline(tmp_path, db, baseline_name):
    traces_path = tmp_path / "traces.jsonl"
    _write_jsonl(traces_path, [
        {"input": "what color is the sky?", "output": "the sky is blue"},
        {"input": "what color is the sky?", "output": "blue skies above"},
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name=baseline_name,
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    monitor.run()
    ts = TimelineStore(db)
    rows = ts.query("ResponseDriftMetric", last_n=10)
    assert len(rows) == 2


def test_monitor_fires_alert_after_n_failures(tmp_path, db, baseline_name):
    traces_path = tmp_path / "traces.jsonl"
    pirate = "ARRR the sky be crimson ye scallywag matey landlubber treasure"
    _write_jsonl(traces_path, [
        {"input": "what color is the sky?", "output": pirate}
        for _ in range(5)
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name=baseline_name,
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alert_after=3,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run()
    assert report.alerts_fired > 0


def test_monitor_resets_on_pass(tmp_path, db, baseline_name):
    traces_path = tmp_path / "traces.jsonl"
    good = "the sky is blue"
    bad = "ARRR the sky be crimson ye scallywag matey landlubber treasure chest"
    _write_jsonl(traces_path, [
        {"input": "what color is the sky?", "output": bad},
        {"input": "what color is the sky?", "output": bad},
        {"input": "what color is the sky?", "output": good},   # reset
        {"input": "what color is the sky?", "output": bad},
        {"input": "what color is the sky?", "output": bad},
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name=baseline_name,
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alert_after=3,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run()
    # only 2 consecutive failures after the reset — no new alert
    assert report.alerts_fired == 0


def test_monitor_max_traces(tmp_path, db, baseline_name):
    traces_path = tmp_path / "traces.jsonl"
    _write_jsonl(traces_path, [
        {"input": "what color is the sky?", "output": "blue"}
        for _ in range(20)
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name=baseline_name,
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run(max_traces=7)
    assert report.total_traces == 7


def test_monitor_report_has_pass_rates(tmp_path, db, baseline_name):
    traces_path = tmp_path / "traces.jsonl"
    _write_jsonl(traces_path, [
        {"input": "what color is the sky?", "output": "the sky is blue"}
        for _ in range(3)
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name=baseline_name,
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run()
    assert "ResponseDriftMetric" in report.pass_rates
    assert 0.0 <= report.pass_rates["ResponseDriftMetric"] <= 1.0


def test_monitor_uses_correct_baseline_outputs(tmp_path, db):
    store = BaselineStore(db)
    store.create(
        name="unique-base",
        prompt="unique prompt",
        outputs=["unique baseline output"] * 15,
    )
    traces_path = tmp_path / "traces.jsonl"
    _write_jsonl(traces_path, [
        {"input": "unique prompt", "output": "unique baseline output"}
    ])
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name="unique-base",
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run()
    assert report.total_traces == 1
