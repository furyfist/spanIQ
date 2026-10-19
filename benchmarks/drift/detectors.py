"""Detectors the drift benchmark can replay.

legacy is spanIQ's pre-0.5 approach: every component output is scored on its
own against the baseline outputs, and an incident starts when a metric fails
for alert_after traces in a row. It uses OutputStabilityMetric only, because
the embedding metrics need a model download.
"""

from __future__ import annotations

from spaniq.core.test_case import LLMTestCase
from spaniq.drift import WindowDriftDetector
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.monitor.trace import Trace
from spaniq.signals import OUTPUT_UNIT, trace_signals


def _component_outputs(record: dict) -> dict[str, str]:
    return {c["name"]: c["output"] for c in record.get("components") or []} or {
        "default": record["output"]
    }


class LegacyStreakDetector:
    name = "legacy"

    def __init__(self, alert_after: int = 3) -> None:
        self.alert_after = alert_after
        self._baseline: dict[str, list[str]] = {}
        self._streak: dict[str, int] = {}
        self._in_incident = False
        self._root: str | None = None

    def fit(self, baseline: list[dict]) -> None:
        for record in baseline:
            for name, output in _component_outputs(record).items():
                self._baseline.setdefault(name, []).append(output)

    def update(self, record: dict) -> bool:
        fired: list[str] = []
        for name, output in _component_outputs(record).items():
            if name not in self._baseline or not output.strip():
                continue
            metric = OutputStabilityMetric()
            metric.measure(
                LLMTestCase(
                    input=record["input"] or "-",
                    actual_output=output,
                    baseline_outputs=self._baseline[name],
                )
            )
            if metric.is_successful():
                self._streak[name] = 0
            else:
                self._streak[name] = self._streak.get(name, 0) + 1
                if self._streak[name] == self.alert_after:
                    fired.append(name)
        if not any(self._streak.values()):
            self._in_incident = False
        if fired and not self._in_incident:
            self._in_incident = True
            self._root = fired[0]
            return True
        return False

    def root_cause(self) -> str | None:
        return self._root


class WindowSignalDetector:
    """spanIQ 0.5: per-unit signal windows compared with the baseline (spaniq.drift).

    The final output and every component get their own WindowDriftDetector. The
    blamed component is the most upstream one whose window drifted, since a
    broken retriever drags the generation step along with it.
    """

    name = "window"

    def __init__(self, window: int = 100, stride: int = 10) -> None:
        self.window = window
        self.stride = stride
        self._detectors: dict[str, WindowDriftDetector] = {}
        self._order: list[str] = []
        self._in_incident = False
        self._root: str | None = None

    def fit(self, baseline: list[dict]) -> None:
        per_unit: dict[str, list[dict]] = {}
        for record in baseline:
            trace = Trace.from_dict(record)
            for unit, signals in trace_signals(trace).items():
                per_unit.setdefault(unit, []).append(signals)
            for c in trace.components or []:
                if c.name not in self._order:
                    self._order.append(c.name)
        self._detectors = {
            unit: WindowDriftDetector(rows, window=self.window, stride=self.stride)
            for unit, rows in per_unit.items()
        }

    def update(self, record: dict) -> bool:
        started = []
        for unit, signals in trace_signals(Trace.from_dict(record)).items():
            detector = self._detectors.get(unit)
            if detector is not None and detector.update(signals) is not None:
                started.append(unit)
        active = any(d.in_incident for d in self._detectors.values())
        if started and not self._in_incident:
            self._in_incident = True
            drifted = [u for u, d in self._detectors.items() if d.in_incident]
            self._root = next((u for u in self._order if u in drifted), None) or (
                OUTPUT_UNIT if OUTPUT_UNIT in drifted else None
            )
            return True
        if not active:
            self._in_incident = False
        return False

    def root_cause(self) -> str | None:
        return self._root


DETECTORS = {"legacy": LegacyStreakDetector, "window": WindowSignalDetector}
