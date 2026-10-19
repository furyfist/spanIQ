from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from rich.console import Console

console = Console(highlight=False)
log = logging.getLogger(__name__)


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
    """Fires one alert when a metric fails for N consecutive traces.

    A failure streak is one incident: it alerts once when the streak reaches
    alert_after and stays quiet until the metric passes again. Set repeat_every
    to re-alert every that many further failures during a long incident.
    """

    alert_after: int = 3
    alerts_path: str = "alerts.jsonl"
    alerts: list[Alert] = field(default_factory=list)
    db_path: str | None = None
    repeat_every: int | None = None

    def __post_init__(self) -> None:
        self._consecutive: dict[str, int] = {}
        self._store = None

    def check(self, metric_results: list, trace_id: str, timestamp: str) -> None:
        """
        Evaluate metric results for a single trace.

        metric_results: list of objects with .metric_name, .score, .threshold, .passed
        """
        for mr in metric_results:
            name = mr.metric_name
            if not mr.passed:
                count = self._consecutive.get(name, 0) + 1
                self._consecutive[name] = count
                if self._should_fire(count):
                    self._fire(mr, trace_id, timestamp)
            else:
                self._consecutive[name] = 0

    def _should_fire(self, count: int) -> bool:
        if count == self.alert_after:
            return True
        if self.repeat_every and count > self.alert_after:
            return (count - self.alert_after) % self.repeat_every == 0
        return False

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
        console.print(f"[bold red]{alert.message}[/bold red]")
        self._append_jsonl(alert)
        if self.db_path:
            self._append_sqlite(alert)

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

    def _append_sqlite(self, alert: Alert) -> None:
        # The JSONL line is already written, so a database failure is logged
        # rather than allowed to stop the monitor mid-stream.
        try:
            if self._store is None:
                from spaniq.monitor.timeline_store import TimelineStore

                self._store = TimelineStore(self.db_path)
            self._store.record_alert(
                timestamp=alert.timestamp,
                metric_name=alert.metric_name,
                score=alert.score,
                threshold=alert.threshold,
                message=alert.message,
                consecutive_count=alert.consecutive_count,
            )
        except Exception:
            log.warning("could not write alert to %s", self.db_path, exc_info=True)
