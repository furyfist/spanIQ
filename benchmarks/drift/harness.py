"""Replay scenario streams through detectors and score them.

A detector sees the baseline traces once (fit), then the stream one trace at a
time (update). update returns True on the trace where an incident starts. After
the first alarm past the change point the harness asks it which component it
blames (root_cause).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from benchmarks.drift.scenarios import Item, Scenario, build_stream


class StreamDetector(Protocol):
    name: str

    def fit(self, baseline: list[dict]) -> None: ...

    def update(self, record: dict) -> bool: ...

    def root_cause(self) -> str | None: ...


@dataclass
class ScenarioResult:
    detector: str
    scenario: str
    seed: int
    n_stream: int
    change_at: int | None
    first_alarm: int | None
    delay: int | None
    false_alarms: int
    pre_change_traces: int
    expected_root: str | None
    found_root: str | None


def run_scenario(
    make_detector: Callable[[], StreamDetector],
    scenario: Scenario,
    items: list[Item],
    seed: int,
    n_baseline: int = 100,
    n_pre: int = 200,
    n_post: int = 300,
) -> ScenarioResult:
    baseline, stream, change_at = build_stream(scenario, items, seed, n_baseline, n_pre, n_post)
    return replay(make_detector(), scenario, seed, baseline, stream, change_at)


def replay(
    detector: StreamDetector,
    scenario: Scenario,
    seed: int,
    baseline: list[dict],
    stream: list[dict],
    change_at: int | None,
) -> ScenarioResult:
    detector.fit(baseline)
    false_alarms = 0
    first_alarm = None
    found_root = None
    for i, record in enumerate(stream):
        if not detector.update(record):
            continue
        if change_at is None or i < change_at:
            false_alarms += 1
        elif first_alarm is None:
            first_alarm = i
            found_root = detector.root_cause()
            break
    pre = len(stream) if change_at is None else change_at
    return ScenarioResult(
        detector=detector.name,
        scenario=scenario.name,
        seed=seed,
        n_stream=len(stream),
        change_at=change_at,
        first_alarm=first_alarm,
        delay=None if first_alarm is None else first_alarm - change_at,
        false_alarms=false_alarms,
        pre_change_traces=pre,
        expected_root=scenario.root,
        found_root=found_root,
    )


def summarize(results: list[ScenarioResult]) -> list[dict]:
    """One row per (detector, scenario): detection rate, median delay, false alarms, root accuracy."""
    groups: dict[tuple[str, str], list[ScenarioResult]] = {}
    for r in results:
        groups.setdefault((r.detector, r.scenario), []).append(r)
    rows = []
    for (detector, scenario), rs in groups.items():
        changed = rs[0].change_at is not None
        detected = [r for r in rs if r.first_alarm is not None]
        pre_total = sum(r.pre_change_traces for r in rs)
        row = {
            "detector": detector,
            "scenario": scenario,
            "runs": len(rs),
            "detection_rate": len(detected) / len(rs) if changed else None,
            "median_delay": statistics.median(r.delay for r in detected) if detected else None,
            "false_alarms_per_1k": 1000 * sum(r.false_alarms for r in rs) / pre_total,
            "root_accuracy": (
                sum(r.found_root == r.expected_root for r in detected) / len(detected)
                if detected and rs[0].expected_root
                else None
            ),
        }
        rows.append(row)
    return rows


def judge_sampling_delay(rate: float, p_flag: float, needed: int = 3) -> float:
    """Expected traces after a change before a sampled judge has flagged `needed` outputs.

    A model, not a measurement: it assumes the judge flags a changed output with
    probability p_flag and that `needed` flags are required to call an incident,
    mirroring alert_after=3.
    """
    if rate <= 0 or p_flag <= 0:
        return float("inf")
    return needed / (rate * p_flag)


def judge_cost_per_1k(rate: float, usd_per_call: float) -> float:
    return 1000 * rate * usd_per_call
