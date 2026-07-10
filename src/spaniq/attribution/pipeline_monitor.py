from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console

from spaniq.attribution.changepoint.cusum import CusumState, cusum_update
from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.timeline_store import TimelineStore
from spaniq.monitor.trace import Trace

console = Console()

_WARMUP_DEFAULT = 20
_WINDOW_DEFAULT = 20
_CUSUM_STRIDE_DEFAULT = 10
_CUSUM_K_FACTOR = 0.5
_CUSUM_H_DEFAULT = 5.0


def _default_span(trace: Trace) -> ComponentSpan:
    return ComponentSpan(
        name="default",
        kind=ComponentKind.DEFAULT,
        output=trace.output,
    )


@dataclass
class ComponentOnlineState:
    window: deque = field(default_factory=deque)
    warmup_outputs: list[str] = field(default_factory=list)
    baseline_outputs: list[str] | None = None
    cusum_states: dict[str, CusumState] = field(default_factory=dict)
    cusum_mu0: dict[str, float] = field(default_factory=dict)
    cusum_k: dict[str, float] = field(default_factory=dict)
    trace_count: int = 0
    online_alarms: dict[str, int] = field(default_factory=dict)


@dataclass
class PipelineReport:
    pipeline_name: str
    total_traces: int
    duration_seconds: float
    components_seen: list[str]
    online_alarms: dict[str, dict[str, int]]
    pass_rates: dict[str, dict[str, float]]


class PipelineMonitor:
    """Per-component monitoring for multi-step LLM pipelines."""

    def __init__(
        self,
        pipeline_name: str,
        collector: BaseCollector,
        metrics: list[BaseMetric] | None = None,
        db_path: str = "spaniq.db",
        window_size: int = _WINDOW_DEFAULT,
        warmup: int = _WARMUP_DEFAULT,
        cusum_stride: int = _CUSUM_STRIDE_DEFAULT,
        cusum_h: float = _CUSUM_H_DEFAULT,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.collector = collector
        self.db_path = db_path
        self.window_size = window_size
        self.warmup = warmup
        self.cusum_stride = cusum_stride
        self.cusum_h = cusum_h

        self.timeline_store = TimelineStore(db_path)
        self.baseline_store = BaselineStore(db_path)

        self.metrics: list[BaseMetric] = metrics or [
            ResponseDriftMetric(),
            SemanticSimilarityMetric(),
            OutputStabilityMetric(),
        ]

        self._state: dict[str, ComponentOnlineState] = {}
        self._pass_counts: dict[str, dict[str, int]] = {}
        self._total_counts: dict[str, dict[str, int]] = {}

    def _get_state(self, component_name: str) -> ComponentOnlineState:
        if component_name not in self._state:
            self._state[component_name] = ComponentOnlineState(
                window=deque(maxlen=self.window_size)
            )
            self._pass_counts[component_name] = {}
            self._total_counts[component_name] = {}
        return self._state[component_name]

    def _process_component(
        self,
        trace: Trace,
        span: ComponentSpan,
    ) -> None:
        state = self._get_state(span.name)
        state.trace_count += 1

        if state.baseline_outputs is None:
            state.warmup_outputs.append(span.output)
            if len(state.warmup_outputs) >= self.warmup:
                state.baseline_outputs = list(state.warmup_outputs)
            return

        if not span.output or not span.output.strip():
            self._record_side_series(trace, span, state)
            return

        state.window.append(span.output)
        if len(state.window) < 2:
            return

        baseline_outputs = state.baseline_outputs
        tc = LLMTestCase(
            input=trace.input,
            actual_output=span.output,
            baseline_outputs=baseline_outputs,
        )

        for metric in self.metrics:
            score = metric.measure(tc)
            passed = metric.is_successful()
            self.timeline_store.record(
                trace_id=trace.trace_id,
                baseline_id=f"{self.pipeline_name}/{span.name}",
                metric_name=metric.name,
                score=score,
                threshold=metric.threshold,
                passed=passed,
                timestamp=trace.timestamp,
                component=span.name,
            )
            comp_pass = self._pass_counts[span.name]
            comp_total = self._total_counts[span.name]
            comp_total[metric.name] = comp_total.get(metric.name, 0) + 1
            if passed:
                comp_pass[metric.name] = comp_pass.get(metric.name, 0) + 1

            self._update_cusum(span.name, metric.name, score, state)

        self._record_side_series(trace, span, state)

    def _update_cusum(
        self,
        component: str,
        metric_name: str,
        score: float,
        state: ComponentOnlineState,
    ) -> None:
        if state.trace_count % self.cusum_stride != 0:
            return
        key = metric_name
        if key not in state.cusum_states:
            state.cusum_states[key] = CusumState()
            state.cusum_mu0[key] = score
            state.cusum_k[key] = _CUSUM_K_FACTOR * 0.05
        cusum_state = state.cusum_states[key]
        if cusum_state.alarm_index is not None:
            return
        mu0 = state.cusum_mu0[key]
        k = state.cusum_k[key]
        new_state = cusum_update(cusum_state, score, mu0=mu0, k=k, h=self.cusum_h)
        state.cusum_states[key] = new_state
        if new_state.alarm_index is not None and key not in state.online_alarms:
            state.online_alarms[key] = state.trace_count

    def _record_side_series(
        self,
        trace: Trace,
        span: ComponentSpan,
        state: ComponentOnlineState,
    ) -> None:
        baseline_id = f"{self.pipeline_name}/{span.name}"
        ts = trace.timestamp
        if span.latency_ms is not None:
            self.timeline_store.record(
                trace_id=trace.trace_id,
                baseline_id=baseline_id,
                metric_name="latency_ms",
                score=span.latency_ms,
                threshold=0.0,
                passed=True,
                timestamp=ts,
                component=span.name,
            )
        output_chars = len(span.output)
        self.timeline_store.record(
            trace_id=trace.trace_id,
            baseline_id=baseline_id,
            metric_name="output_chars",
            score=float(output_chars),
            threshold=0.0,
            passed=True,
            timestamp=ts,
            component=span.name,
        )

    def run(self, max_traces: int | None = None) -> PipelineReport:
        start = time.perf_counter()
        processed = 0
        for trace in self.collector.collect():
            spans = trace.components or [_default_span(trace)]
            for span in spans:
                self._process_component(trace, span)
            processed += 1
            if max_traces and processed >= max_traces:
                break
        duration = time.perf_counter() - start

        alarms = {comp: dict(state.online_alarms) for comp, state in self._state.items()}
        pass_rates = {
            comp: {
                m: self._pass_counts[comp].get(m, 0) / total
                for m, total in self._total_counts[comp].items()
                if total > 0
            }
            for comp in self._total_counts
        }
        console.print(
            f"[green]pipeline monitor complete[/green] — "
            f"{processed} traces, {len(self._state)} component(s)"
        )
        return PipelineReport(
            pipeline_name=self.pipeline_name,
            total_traces=processed,
            duration_seconds=duration,
            components_seen=list(self._state.keys()),
            online_alarms=alarms,
            pass_rates=pass_rates,
        )
