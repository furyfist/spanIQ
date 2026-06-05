from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from spaniq.core.evaluate import evaluate
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
from spaniq.monitor.alerting import Alert, AlertEngine
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.timeline_store import TimelineStore
from spaniq.monitor.trace import Trace

console = Console()


@dataclass
class MonitorReport:
    total_traces: int
    duration_seconds: float
    alerts_fired: int
    alerts: list[Alert]
    pass_rates: dict[str, float]  # metric_name -> pass rate over the run


class Monitor:
    """Continuous evaluation of LLM traces against a stored baseline."""

    def __init__(
        self,
        baseline_name: str,
        collector: BaseCollector,
        metrics: list[BaseMetric] | None = None,
        db_path: str = "spaniq.db",
        alert_after: int = 3,
        alerts_path: str = "alerts.jsonl",
    ) -> None:
        self.baseline_store = BaselineStore(db_path)
        self.timeline_store = TimelineStore(db_path)
        self.alert_engine = AlertEngine(alert_after=alert_after, alerts_path=alerts_path)
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
        )

    def _process_trace(self, trace: Trace) -> None:
        tc = LLMTestCase(
            input=trace.input,
            actual_output=trace.output,
            baseline_outputs=self._baseline_outputs,
        )
        result = evaluate([tc], self.metrics, verbose=False)
        mr_list = result.test_case_results[0].metric_results

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
