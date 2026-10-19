from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console

from spaniq.attribution.changepoint.cusum import CusumState, cusum_update
from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.attribution.drift_attribution import DriftAttribution, attribute_drift
from spaniq.core.test_case import LLMTestCase
from spaniq.drift.detector import WindowDriftDetector
from spaniq.metrics.base import BaseMetric
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
from spaniq.monitor.alerting import AlertEngine
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.timeline_store import TimelineStore
from spaniq.monitor.trace import Trace
from spaniq.signals.extract import OUTPUT_UNIT, trace_signals

console = Console()

_WARMUP_DEFAULT = 20
_CUSUM_STRIDE_DEFAULT = 10
_CUSUM_CALIBRATION_DEFAULT = 10
_CUSUM_K_FACTOR = 0.5
_CUSUM_H_DEFAULT = 5.0
# Floor for the calibrated score sigma, so a perfectly constant metric does not
# make the first tiny wobble an alarm.
_CUSUM_MIN_SIGMA = 0.01


def _default_span(trace: Trace) -> ComponentSpan:
    return ComponentSpan(
        name="default",
        kind=ComponentKind.DEFAULT,
        output=trace.output,
    )


@dataclass
class ComponentOnlineState:
    warmup_outputs: list[str] = field(default_factory=list)
    baseline_outputs: list[str] | None = None
    cusum_states: dict[str, CusumState] = field(default_factory=dict)
    cusum_calibration: dict[str, list[float]] = field(default_factory=dict)
    cusum_mu0: dict[str, float] = field(default_factory=dict)
    cusum_k: dict[str, float] = field(default_factory=dict)
    cusum_h: dict[str, float] = field(default_factory=dict)
    trace_count: int = 0
    scored_count: int = 0
    online_alarms: dict[str, int] = field(default_factory=dict)
    warmup_signals: list[dict[str, float]] = field(default_factory=list)
    drift_detector: WindowDriftDetector | None = None
    drift_alarms: list[int] = field(default_factory=list)


@dataclass
class PipelineReport:
    pipeline_name: str
    total_traces: int
    duration_seconds: float
    components_seen: list[str]
    online_alarms: dict[str, dict[str, int]]
    pass_rates: dict[str, dict[str, float]]
    drift_alarms: dict[str, list[int]] = field(default_factory=dict)
    drift_attributions: list[DriftAttribution] = field(default_factory=list)


class PipelineMonitor:
    """Per-component monitoring for multi-step LLM pipelines.

    Each component's first `warmup` outputs become its frozen baseline, and
    every later output is scored against that baseline. The baseline is not a
    rolling window on purpose: a window that follows recent outputs would absorb
    a slow drift instead of flagging it.

    Online CUSUM runs per (component, metric). Its reference mean and sigma are
    estimated from the first `cusum_calibration` scored traces, the slack is
    0.5 sigma, and cusum_h is the alarm threshold in sigma units.

    Each component's signals are stored as signal.<name> series (for
    `spaniq attribute --signals`) and a window of them is compared with the
    warmup traces' signals; a shift raises one drift alert per incident and is
    listed in PipelineReport.drift_alarms by trace count.
    """

    def __init__(
        self,
        pipeline_name: str,
        collector: BaseCollector,
        metrics: list[BaseMetric] | None = None,
        db_path: str = "spaniq.db",
        warmup: int = _WARMUP_DEFAULT,
        cusum_stride: int = _CUSUM_STRIDE_DEFAULT,
        cusum_h: float = _CUSUM_H_DEFAULT,
        cusum_calibration: int = _CUSUM_CALIBRATION_DEFAULT,
        drift_window: int | None = 100,
        drift_stride: int = 25,
        record_signals: bool = True,
        alerts_path: str | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.collector = collector
        self.db_path = db_path
        self.warmup = warmup
        self.cusum_stride = cusum_stride
        self.cusum_h = cusum_h
        self.cusum_calibration = cusum_calibration

        self.drift_window = drift_window
        self.drift_stride = drift_stride
        self.record_signals = record_signals

        self.timeline_store = TimelineStore(db_path)
        self.baseline_store = BaselineStore(db_path)
        self.alert_engine = AlertEngine(alerts_path=alerts_path, db_path=db_path)

        self.metrics: list[BaseMetric] = metrics or [
            ResponseDriftMetric(),
            SemanticSimilarityMetric(),
            OutputStabilityMetric(),
        ]

        self._state: dict[str, ComponentOnlineState] = {}
        self._order: list[str] = []
        self._recent: deque[dict[str, dict[str, float]]] = deque(maxlen=drift_window or 1)
        self._attributions: list[DriftAttribution] = []
        self._incident_open = False
        self._pass_counts: dict[str, dict[str, int]] = {}
        self._total_counts: dict[str, dict[str, int]] = {}

    def _get_state(self, component_name: str) -> ComponentOnlineState:
        if component_name not in self._state:
            self._state[component_name] = ComponentOnlineState()
            self._pass_counts[component_name] = {}
            self._total_counts[component_name] = {}
        return self._state[component_name]

    def _process_component(
        self,
        trace: Trace,
        span: ComponentSpan,
        signals: dict[str, float] | None = None,
    ) -> None:
        state = self._get_state(span.name)
        state.trace_count += 1

        if state.baseline_outputs is None:
            state.warmup_outputs.append(span.output)
            if signals is not None:
                state.warmup_signals.append(signals)
            if len(state.warmup_outputs) >= self.warmup:
                state.baseline_outputs = list(state.warmup_outputs)
                if self.drift_window and len(state.warmup_signals) >= 5:
                    state.drift_detector = WindowDriftDetector(
                        state.warmup_signals,
                        window=self.drift_window,
                        stride=self.drift_stride,
                    )
            return

        if signals is not None:
            self._track_signals(trace, span, state, signals)

        if not span.output or not span.output.strip():
            self._record_side_series(trace, span, state)
            return

        state.scored_count += 1
        tc = LLMTestCase(
            input=trace.input,
            actual_output=span.output,
            baseline_outputs=state.baseline_outputs,
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

    def _track_signals(
        self,
        trace: Trace,
        span: ComponentSpan,
        state: ComponentOnlineState,
        signals: dict[str, float],
    ) -> None:
        if self.record_signals:
            self.timeline_store.record_signals(
                trace.trace_id,
                f"{self.pipeline_name}/{span.name}",
                signals,
                trace.timestamp,
                component=span.name,
            )
        if state.drift_detector is None:
            return
        report = state.drift_detector.update(signals)
        if report is None:
            return
        state.drift_alarms.append(state.trace_count)
        alert = self.alert_engine.fire_drift(
            span.name, report, trace.trace_id, trace.timestamp, component=span.name
        )
        if self._incident_open:
            return
        # first component to trip in this incident: work out which one to blame
        self._incident_open = True
        windows = {
            name: [t[name] for t in self._recent if name in t]
            for name, s in self._state.items()
            if s.warmup_signals
        }
        baselines = {name: s.warmup_signals for name, s in self._state.items() if s.warmup_signals}
        attribution = attribute_drift(span.name, report, windows, baselines, self._order)
        self._attributions.append(attribution)
        alert.message += f" | {attribution.verdict}"
        console.print(f"[yellow]attribution:[/yellow] {attribution.verdict}")

    def _close_finished_incident(self) -> None:
        if self._incident_open and not any(
            s.drift_detector is not None and s.drift_detector.in_incident
            for s in self._state.values()
        ):
            self._incident_open = False

    def _update_cusum(
        self,
        component: str,
        metric_name: str,
        score: float,
        state: ComponentOnlineState,
    ) -> None:
        key = metric_name
        if key not in state.cusum_states:
            calibration = state.cusum_calibration.setdefault(key, [])
            calibration.append(score)
            if len(calibration) < self.cusum_calibration:
                return
            sigma = max(float(np.std(calibration)), _CUSUM_MIN_SIGMA)
            state.cusum_mu0[key] = float(np.mean(calibration))
            state.cusum_k[key] = _CUSUM_K_FACTOR * sigma
            state.cusum_h[key] = self.cusum_h * sigma
            state.cusum_states[key] = CusumState()
            return
        # stride counts scored traces only, so warmup does not shift the phase
        if state.scored_count % self.cusum_stride != 0:
            return
        cusum_state = state.cusum_states[key]
        if cusum_state.alarm_index is not None:
            return
        new_state = cusum_update(
            cusum_state,
            score,
            mu0=state.cusum_mu0[key],
            k=state.cusum_k[key],
            h=state.cusum_h[key],
        )
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
            units = trace_signals(trace)
            for span in spans:
                if span.name not in self._order:
                    self._order.append(span.name)
            self._recent.append({s.name: units.get(s.name) or units[OUTPUT_UNIT] for s in spans})
            self._close_finished_incident()
            for span in spans:
                signals = units.get(span.name)
                if signals is None and span.name == "default":
                    signals = units[OUTPUT_UNIT]
                self._process_component(trace, span, signals)
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
            drift_alarms={
                comp: list(state.drift_alarms)
                for comp, state in self._state.items()
                if state.drift_alarms
            },
            drift_attributions=list(self._attributions),
        )
