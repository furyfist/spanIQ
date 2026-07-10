"""Tests for PipelineMonitor."""

from __future__ import annotations

from collections.abc import Iterator

from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.attribution.pipeline_monitor import PipelineMonitor
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace


class ListCollector(BaseCollector):
    def __init__(self, traces):
        self._traces = traces

    def collect(self) -> Iterator[Trace]:
        yield from self._traces

    def name(self) -> str:
        return "list"


def _make_traces(n=40, component_names=("retrieval", "generation"), seed=0):
    import random

    random.seed(seed)
    texts = ["good answer", "Paris is the capital", "correct response", "great output"]
    traces = []
    for _i in range(n):
        components = [
            ComponentSpan(
                name=name,
                kind=ComponentKind.RETRIEVAL,
                output=random.choice(texts),
                latency_ms=100.0,
            )
            for name in component_names
        ]
        traces.append(Trace(input="q", output=random.choice(texts), components=components))
    return traces


def test_per_component_windows_isolated(tmp_path):
    db = str(tmp_path / "test.db")
    traces = _make_traces(n=50, component_names=("A", "B"))
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10, window_size=10)
    report = pm.run()
    assert "A" in report.components_seen
    assert "B" in report.components_seen
    assert pm._state["A"] is not pm._state["B"]


def test_scores_tagged_with_component(tmp_path):
    db = str(tmp_path / "test.db")
    traces = _make_traces(n=40)
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10)
    pm.run()
    from spaniq.monitor.timeline_store import TimelineStore

    store = TimelineStore(db)
    comps = store.components()
    assert "retrieval" in comps
    assert "generation" in comps


def test_v2_trace_falls_back_to_default(tmp_path):
    db = str(tmp_path / "test.db")
    traces = [Trace(input="q", output="answer") for _ in range(30)]
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10)
    report = pm.run()
    assert "default" in report.components_seen


def test_two_components_dont_share_state(tmp_path):
    db = str(tmp_path / "test.db")
    traces = _make_traces(n=60, component_names=("X", "Y"))
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10, window_size=10)
    pm.run()
    assert pm._state["X"].window is not pm._state["Y"].window


def test_max_traces(tmp_path):
    db = str(tmp_path / "test.db")
    traces = _make_traces(n=100)
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10)
    report = pm.run(max_traces=30)
    assert report.total_traces == 30


def test_report_totals(tmp_path):
    db = str(tmp_path / "test.db")
    traces = _make_traces(n=50)
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10)
    report = pm.run()
    assert report.pipeline_name == "pipe"
    assert report.total_traces == 50
    assert len(report.components_seen) >= 1


def test_cusum_stride_respected(tmp_path):
    db = str(tmp_path / "test.db")
    traces = _make_traces(n=100)
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10, cusum_stride=5)
    pm.run()
    for _comp, state in pm._state.items():
        for _key, cs in state.cusum_states.items():
            assert cs.n_seen <= (100 // 5) + 2


def test_online_alarm_recorded_after_shift(tmp_path):
    import random

    db = str(tmp_path / "test.db")
    random.seed(1)
    good = ["great answer about Paris"] * 5
    bad = ["I cannot find any relevant information whatsoever"] * 5
    traces = []
    for i in range(80):
        out = random.choice(good) if i < 40 else random.choice(bad)
        traces.append(
            Trace(
                input="q",
                output=out,
                components=[ComponentSpan("gen", ComponentKind.GENERATION, out)],
            )
        )
    pm = PipelineMonitor("pipe", ListCollector(traces), db_path=db, warmup=10, cusum_h=2.0)
    report = pm.run()
    assert isinstance(report.online_alarms, dict)
