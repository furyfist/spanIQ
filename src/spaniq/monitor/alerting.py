from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field

from rich.console import Console

_stdout = (
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer")
    else sys.stdout
)
console = Console(file=_stdout, highlight=False)


@dataclass
class Alert:
    metric_name: str
    score: float
    threshold: float
    trace_id: str
    timestamp: str
    message: str
    consecutive_count: int


@dataclass
class AlertEngine:
    """Fires alerts when a metric fails for N consecutive traces."""

    alert_after: int = 3
    alerts_path: str = "alerts.jsonl"
    alerts: list[Alert] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._consecutive: dict[str, int] = {}

    def check(self, metric_results: list, trace_id: str, timestamp: str) -> None:
        """
        Evaluate metric results for a single trace.

        metric_results: list of objects with .metric_name, .score, .threshold, .passed
        """
        for mr in metric_results:
            name = mr.metric_name
            if not mr.passed:
                self._consecutive[name] = self._consecutive.get(name, 0) + 1
                if self._consecutive[name] >= self.alert_after:
                    self._fire(mr, trace_id, timestamp)
            else:
                self._consecutive[name] = 0

    def _fire(self, mr, trace_id: str, timestamp: str) -> None:
        count = self._consecutive[mr.metric_name]
        alert = Alert(
            metric_name=mr.metric_name,
            score=mr.score,
            threshold=mr.threshold,
            trace_id=trace_id,
            timestamp=timestamp,
            message=(
                f"ALERT: {mr.metric_name} crossed threshold "
                f"({mr.score:.4f} vs {mr.threshold}) "
                f"for {count} consecutive traces"
            ),
            consecutive_count=count,
        )
        self.alerts.append(alert)
        console.print(f"[bold red]🔴 {alert.message}[/bold red]")
        self._append_jsonl(alert)

    def _append_jsonl(self, alert: Alert) -> None:
        with open(self.alerts_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "metric_name": alert.metric_name,
                        "score": alert.score,
                        "threshold": alert.threshold,
                        "trace_id": alert.trace_id,
                        "timestamp": alert.timestamp,
                        "message": alert.message,
                        "consecutive_count": alert.consecutive_count,
                    }
                )
                + "\n"
            )
