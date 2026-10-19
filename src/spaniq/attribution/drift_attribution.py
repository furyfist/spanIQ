"""Which component caused a drift incident.

When one component's window trips, the others are re-tested on the same
traces, but only from the point where the change seems to have started: the
tripping window still holds mostly pre-change traces, and an upstream change
with a weaker footprint (a retriever returning the wrong passage) would be
diluted below significance by them. The re-test also uses looser thresholds,
since it confirms a known incident rather than raising a new one.

The blamed component is the most upstream one that shifted. This is temporal
and structural reasoning, not causal proof: a broken retriever and a generation
step that changed at the same moment look the same (see
docs/decisions/temporal-not-causal.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spaniq.drift.detector import EMBEDDING, DriftReport, compare

CONFIRM_ALPHA = 0.01
CONFIRM_MIN_EFFECT = 0.3
MIN_SEGMENT = 5


@dataclass
class DriftAttribution:
    root: str | None
    implicated: list[str]
    change_start: int
    reports: dict[str, DriftReport] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.root is None:
            return "no component shifted"
        rest = [u for u in self.implicated if u != self.root]
        if not rest:
            return f"{self.root} shifted; no other component did"
        return (
            f"{self.root} is the most upstream component that shifted; {', '.join(rest)} followed"
        )


def estimate_change_start(values: list[float]) -> int:
    """Index where the mean of a series most clearly changes (one-split binary segmentation).

    Maximises |mean(after) - mean(before)| * sqrt(k (n - k) / n), the statistic
    behind CUSUM-style single change point estimates. Returns 0 for a flat series.
    """
    x = np.asarray(values, float)
    n = len(x)
    if n < 2 * MIN_SEGMENT or np.ptp(x) == 0:
        return 0
    cumulative = np.cumsum(x)
    total = cumulative[-1]
    best_k, best = 0, 0.0
    for k in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        before = cumulative[k - 1] / k
        after = (total - cumulative[k - 1]) / (n - k)
        stat = abs(after - before) * np.sqrt(k * (n - k) / n)
        if stat > best:
            best_k, best = k, float(stat)
    return best_k


def attribute_drift(
    trigger: str,
    trigger_report: DriftReport,
    windows: dict[str, list[dict[str, float]]],
    baselines: dict[str, list[dict[str, float]]],
    order: list[str],
    alpha: float = CONFIRM_ALPHA,
    min_effect: float = CONFIRM_MIN_EFFECT,
) -> DriftAttribution:
    """Blame the most upstream unit whose post-change traces differ from its baseline.

    windows hold the same recent traces for every unit, oldest first. order
    lists components upstream first; units not in it (the final output) rank last.
    """
    strongest = next(
        (s.signal for s in trigger_report.drifted_signals if s.signal != EMBEDDING), None
    )
    start = 0
    if strongest is not None:
        series = [r[strongest] for r in windows.get(trigger, []) if strongest in r]
        start = estimate_change_start(series)

    reports: dict[str, DriftReport] = {}
    implicated: list[str] = []
    for unit, rows in windows.items():
        segment = rows[start:]
        if unit == trigger:
            reports[unit] = trigger_report
            implicated.append(unit)
            continue
        if len(segment) < MIN_SEGMENT or unit not in baselines:
            continue
        report = compare(baselines[unit], segment, alpha=alpha, min_effect=min_effect)
        reports[unit] = report
        if report.drifted:
            implicated.append(unit)

    rank = {u: i for i, u in enumerate(order)}
    implicated.sort(key=lambda u: rank.get(u, len(order)))
    return DriftAttribution(
        root=implicated[0] if implicated else None,
        implicated=implicated,
        change_start=start,
        reports=reports,
    )
