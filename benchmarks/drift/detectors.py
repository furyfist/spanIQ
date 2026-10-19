"""Detectors the drift benchmark can replay.

legacy is spanIQ's pre-0.5 approach: every component output is scored on its
own against the baseline outputs, and an incident starts when a metric fails
for alert_after traces in a row. It uses OutputStabilityMetric only, because
the embedding metrics need a model download.
"""

from __future__ import annotations

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.output_stability import OutputStabilityMetric


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


DETECTORS = {"legacy": LegacyStreakDetector}
