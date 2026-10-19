from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from spaniq.core.eval_result import MetricResult
from spaniq.core.test_case import LLMTestCase
from spaniq.drift.detector import DriftReport, WindowDriftDetector
from spaniq.escalation import Candidate, JudgeBudget, Verdict, drift_priorities, escalate, select
from spaniq.escalation.judge import Judge
from spaniq.metrics.base import BaseMetric
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
from spaniq.monitor.alerting import Alert, AlertEngine
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.timeline_store import TimelineStore
from spaniq.monitor.trace import Trace
from spaniq.signals.extract import context_for, extract
from spaniq.signals.text import text_signals

console = Console()


@dataclass
class MonitorReport:
    total_traces: int
    duration_seconds: float
    alerts_fired: int
    alerts: list[Alert]
    pass_rates: dict[str, float]  # metric_name -> pass rate over the run
    drift_incidents: list[DriftReport] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)


class Monitor:
    """Continuous evaluation of LLM traces against a stored baseline.

    Besides scoring each trace with the metrics, every trace's signals are
    stored (as signal.<name> series) and a sliding window of them is compared
    with the baseline outputs' signals; a shift raises one drift alert per
    incident. Set drift_window=None to turn the window check off.

    With a judge, the start of each drift incident sends the judge_per_incident
    traces of the window that moved furthest along the drift (within
    judge_budget, if given) to the judge; verdicts are stored as the "judge" series.
    """

    def __init__(
        self,
        baseline_name: str,
        collector: BaseCollector,
        metrics: list[BaseMetric] | None = None,
        db_path: str = "spaniq.db",
        alert_after: int = 3,
        alerts_path: str = "alerts.jsonl",
        drift_window: int | None = 100,
        drift_stride: int = 25,
        record_signals: bool = True,
        judge: Judge | None = None,
        judge_budget: JudgeBudget | None = None,
        judge_per_incident: int = 5,
    ) -> None:
        self.baseline_store = BaselineStore(db_path)
        self.timeline_store = TimelineStore(db_path)
        self.alert_engine = AlertEngine(
            alert_after=alert_after, alerts_path=alerts_path, db_path=db_path
        )
        self.collector = collector
        self.baseline = self.baseline_store.get_by_name(baseline_name)
        self._baseline_outputs: list[str] = json.loads(self.baseline.outputs)
        self.metrics: list[BaseMetric] = metrics or [
            ResponseDriftMetric(),
            SemanticSimilarityMetric(),
            OutputStabilityMetric(),
        ]
        self._metric_pass_counts: dict[str, int] = {}
        self._metric_total_counts: dict[str, int] = {}
        self.record_signals = record_signals
        self.drift_incidents: list[DriftReport] = []
        self.judge = judge
        self.judge_budget = judge_budget
        self.judge_per_incident = judge_per_incident
        self.verdicts: list[Verdict] = []
        self._recent: deque[tuple[Trace, dict[str, float], str | None]] = deque(
            maxlen=drift_window or 1
        )
        # Baselines store outputs only, so the window compares reference-free signals.
        baseline_signals = [text_signals(o) for o in self._baseline_outputs]
        self._baseline_signals = baseline_signals
        self.drift_detector = (
            WindowDriftDetector(baseline_signals, window=drift_window, stride=drift_stride)
            if drift_window and len(baseline_signals) >= 5
            else None
        )

    def run(self, max_traces: int | None = None) -> MonitorReport:
        """Run the monitor loop. Blocks until collector is exhausted or max_traces reached."""
        start = time.perf_counter()
        processed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"monitoring against baseline '{self.baseline.name}' …", total=None
            )
            for trace in self.collector.collect():
                self._process_trace(trace)
                processed += 1
                progress.update(task, description=f"processed {processed} traces …")
                if max_traces and processed >= max_traces:
                    break

        duration = time.perf_counter() - start
        pass_rates = {
            name: (self._metric_pass_counts.get(name, 0) / total)
            for name, total in self._metric_total_counts.items()
            if total > 0
        }
        console.print(
            f"[green]monitor complete[/green] — "
            f"{processed} traces in {duration:.1f}s, "
            f"{len(self.alert_engine.alerts)} alert(s) fired"
        )
        return MonitorReport(
            total_traces=processed,
            duration_seconds=duration,
            alerts_fired=len(self.alert_engine.alerts),
            alerts=self.alert_engine.alerts,
            pass_rates=pass_rates,
            drift_incidents=list(self.drift_incidents),
            verdicts=list(self.verdicts),
        )

    def _process_trace(self, trace: Trace) -> None:
        tc = LLMTestCase(
            input=trace.input,
            actual_output=trace.output,
            baseline_outputs=self._baseline_outputs,
            retrieval_context=[context] if (context := context_for(trace)) else None,
        )
        mr_list = []
        for metric in self.metrics:
            score = metric.measure(tc)
            mr_list.append(
                MetricResult(
                    metric_name=metric.name,
                    score=score,
                    threshold=metric.threshold,
                    passed=metric.is_successful(),
                    reason=metric.reason,
                )
            )

        for mr in mr_list:
            self.timeline_store.record(
                trace_id=trace.trace_id,
                baseline_id=self.baseline.id,
                metric_name=mr.metric_name,
                score=mr.score,
                threshold=mr.threshold,
                passed=mr.passed,
                timestamp=trace.timestamp,
            )
            self._metric_total_counts[mr.metric_name] = (
                self._metric_total_counts.get(mr.metric_name, 0) + 1
            )
            if mr.passed:
                self._metric_pass_counts[mr.metric_name] = (
                    self._metric_pass_counts.get(mr.metric_name, 0) + 1
                )

        self.alert_engine.check(mr_list, trace.trace_id, trace.timestamp)

        context = tc.retrieval_context[0] if tc.retrieval_context else None
        signals = extract(trace.output, context)
        self._recent.append((trace, signals, context))
        if self.record_signals:
            self.timeline_store.record_signals(
                trace.trace_id, self.baseline.id, signals, trace.timestamp
            )
        if self.drift_detector is not None:
            report = self.drift_detector.update(signals)
            if report is not None:
                self.drift_incidents.append(report)
                self.alert_engine.fire_drift("output", report, trace.trace_id, trace.timestamp)
                if self.judge is not None:
                    self._escalate(report)

    def _escalate(self, report: DriftReport) -> None:
        window = list(self._recent)
        priorities = drift_priorities([s for _, s, _ in window], self._baseline_signals, report)
        candidates = [
            Candidate(t.trace_id, t.input, t.output, ctx, priority, tuple(reasons))
            for (t, _, ctx), (priority, reasons) in zip(window, priorities, strict=True)
        ]
        chosen = select(candidates, self.judge_per_incident)
        verdicts = escalate(
            chosen,
            self.judge,
            self.judge_budget,
            timeline=self.timeline_store,
            baseline_id=self.baseline.id,
            timestamp=window[-1][0].timestamp,
        )
        self.verdicts += verdicts
        failed = sum(not v.passed for v in verdicts)
        if verdicts:
            console.print(
                f"[yellow]judge:[/yellow] {failed} of {len(verdicts)} escalated trace(s) judged bad"
            )
