"""End-to-end integration test: full pipeline from baseline creation to alert detection."""
from __future__ import annotations

import json

from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.file import FileCollector
from spaniq.monitor.monitor import Monitor
from spaniq.monitor.timeline_store import TimelineStore
from spaniq.metrics.response_drift import ResponseDriftMetric


def _write_jsonl(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_full_monitoring_pipeline(tmp_path):
    """Create baseline → write traces → monitor → detect drift → alert."""
    db_path = str(tmp_path / "e2e.db")

    # 1. create baseline with clearly on-topic outputs
    store = BaselineStore(db_path)
    store.create(
        name="e2e-baseline",
        prompt="what color is the sky?",
        outputs=["the sky is blue"] * 20,
    )

    # 2. write 10 normal + 10 clearly drifted traces
    traces_path = tmp_path / "traces.jsonl"
    normal = [
        {"input": "what color is the sky?", "output": "the sky is blue"}
        for _ in range(10)
    ]
    drifted = [
        {
            "input": "what color is the sky?",
            "output": "ARRR the sky be crimson scallywag matey landlubber treasure pirate",
        }
        for _ in range(10)
    ]
    _write_jsonl(traces_path, normal + drifted)

    # 3. run monitor
    collector = FileCollector(str(traces_path))
    monitor = Monitor(
        baseline_name="e2e-baseline",
        collector=collector,
        metrics=[ResponseDriftMetric()],
        db_path=db_path,
        alert_after=3,
        alerts_path=str(tmp_path / "alerts.jsonl"),
    )
    report = monitor.run()

    # 4. basic report assertions
    assert report.total_traces == 20
    assert report.alerts_fired > 0
    assert report.alerts[0].metric_name == "ResponseDriftMetric"

    # 5. timeline is fully populated
    timeline = TimelineStore(db_path)
    rows = timeline.query("ResponseDriftMetric", last_n=20)
    assert len(rows) == 20

    # 6. first 10 (normal) should have a higher pass rate than last 10 (drifted)
    # rows are DESC by timestamp, so rows[-10:] are the earliest (normal) traces
    # and rows[:10] are the latest (drifted) traces
    drifted_rows = rows[:10]
    normal_rows = rows[10:]
    drifted_pass_rate = sum(r.passed for r in drifted_rows) / 10
    normal_pass_rate = sum(r.passed for r in normal_rows) / 10
    assert normal_pass_rate >= drifted_pass_rate

    # 7. alert was written to JSONL
    alerts_file = tmp_path / "alerts.jsonl"
    assert alerts_file.exists()
    with open(alerts_file) as f:
        alert_entries = [json.loads(line) for line in f if line.strip()]
    assert len(alert_entries) > 0
    assert alert_entries[0]["metric_name"] == "ResponseDriftMetric"
    assert "ALERT" in alert_entries[0]["message"]
