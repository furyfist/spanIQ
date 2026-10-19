"""Window drift detection inside Monitor and PipelineMonitor."""

from __future__ import annotations

import json
from collections.abc import Iterator

from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.attribution.pipeline_monitor import PipelineMonitor
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.collectors.file import FileCollector
from spaniq.monitor.monitor import Monitor
from spaniq.monitor.timeline_store import TimelineStore
from spaniq.monitor.trace import Trace

GOOD = [
    "The sky is blue because air scatters short wavelengths of sunlight.",
    "Blue light scatters more in the atmosphere, so the sky looks blue.",
    "Rayleigh scattering of sunlight makes the daytime sky appear blue.",
    "Short blue wavelengths scatter most, which is why the sky is blue.",
]
REFUSAL = "I'm sorry, I can't help with that."


class ListCollector(BaseCollector):
    def __init__(self, traces):
        self._traces = traces

    def collect(self) -> Iterator[Trace]:
        yield from self._traces

    def name(self) -> str:
        return "list"


def _monitor(tmp_path, outputs, **kwargs):
    db = str(tmp_path / "m.db")
    BaselineStore(db).create(name="base", prompt="why is the sky blue?", outputs=GOOD * 10)
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"trace_id": f"t{i}", "input": "why is the sky blue?", "output": o})
            for i, o in enumerate(outputs)
        ),
        encoding="utf-8",
    )
    monitor = Monitor(
        baseline_name="base",
        collector=FileCollector(str(path)),
        metrics=[OutputStabilityMetric()],
        db_path=db,
        alerts_path=str(tmp_path / "alerts.jsonl"),
        **kwargs,
    )
    return monitor, db


def test_monitor_raises_one_drift_alert_per_incident(tmp_path):
    outputs = [GOOD[i % 4] for i in range(60)] + [REFUSAL] * 120
    monitor, db = _monitor(tmp_path, outputs, drift_window=40, drift_stride=10)
    report = monitor.run()
    assert len(report.drift_incidents) == 1
    assert report.drift_incidents[0].get("refusal").drifted
    drift_alerts = [a for a in report.alerts if a.metric_name == "drift:output"]
    assert len(drift_alerts) == 1
    assert "DRIFT" in drift_alerts[0].message
    stored = TimelineStore(db).query_alerts(last_n=10)
    assert any(a["metric_name"] == "drift:output" for a in stored)


def test_monitor_quiet_without_change(tmp_path):
    monitor, _ = _monitor(tmp_path, [GOOD[i % 4] for i in range(150)], drift_window=40)
    assert monitor.run().drift_incidents == []


def test_monitor_records_signal_series(tmp_path):
    monitor, db = _monitor(tmp_path, GOOD * 3)
    monitor.run()
    rows = TimelineStore(db).query("signal.chars", last_n=100)
    assert len(rows) == 12


def test_monitor_signal_recording_and_window_can_be_disabled(tmp_path):
    monitor, db = _monitor(tmp_path, GOOD * 3, drift_window=None, record_signals=False)
    assert monitor.drift_detector is None
    monitor.run()
    assert TimelineStore(db).query("signal.chars", last_n=100) == []


def _pipeline_traces(n_good, n_bad):
    traces = []
    for i in range(n_good + n_bad):
        broken = i >= n_good
        context = (
            "Unrelated passage about volcanoes."
            if broken
            else "Rayleigh scattering of sunlight by air makes the sky blue."
        )
        answer = GOOD[i % 4]
        traces.append(
            Trace(
                input="why is the sky blue?",
                output=answer,
                trace_id=f"p{i}",
                timestamp=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
                components=[
                    ComponentSpan("retrieval", ComponentKind.RETRIEVAL, context),
                    ComponentSpan("generation", ComponentKind.GENERATION, answer),
                ],
            )
        )
    return traces


def test_pipeline_monitor_drift_alarm_on_broken_retriever(tmp_path):
    db = str(tmp_path / "p.db")
    pm = PipelineMonitor(
        "rag",
        ListCollector(_pipeline_traces(80, 80)),
        metrics=[OutputStabilityMetric()],
        db_path=db,
        warmup=40,
        drift_window=30,
        drift_stride=10,
    )
    report = pm.run()
    assert "retrieval" in report.drift_alarms
    assert report.drift_alarms["retrieval"][0] > 80
    store = TimelineStore(db)
    rows = store.query("signal.question_coverage", component="retrieval", last_n=500)
    assert len(rows) == 120
    assert store.query_alerts(component="retrieval", metric_name="drift:retrieval")


def test_pipeline_monitor_without_components_uses_output_signals(tmp_path):
    db = str(tmp_path / "p.db")
    traces = [Trace(input="q", output=GOOD[i % 4]) for i in range(30)]
    pm = PipelineMonitor(
        "plain", ListCollector(traces), metrics=[OutputStabilityMetric()], db_path=db, warmup=10
    )
    pm.run()
    assert len(TimelineStore(db).query("signal.words", component="default", last_n=100)) == 20


def test_cli_drift_window_flag():
    from spaniq.cli.runner import _build_parser

    args = _build_parser().parse_args(
        ["pipeline", "run", "--name", "x", "--path", "p", "--drift-window", "0"]
    )
    assert args.drift_window == 0
    args = _build_parser().parse_args(["monitor", "run", "--baseline", "b", "--path", "p"])
    assert args.drift_window == 100
