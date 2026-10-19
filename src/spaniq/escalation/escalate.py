"""Spend judge calls where the signals say something moved.

Signals run on every trace for free; an LLM judge is accurate but costs money
per call. Escalation ranks traces by how far they moved in the direction of a
detected drift and sends only the top ones, within a daily budget, to a judge.
An optional exploration share sends a few random traces too, so the judge also
sees ordinary traffic and its scores can be compared against the signals.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from spaniq.escalation.budget import JudgeBudget
from spaniq.escalation.judge import Judge, JudgeError

log = logging.getLogger(__name__)
JUDGE_METRIC = "judge"


@dataclass
class Candidate:
    trace_id: str
    input: str
    output: str
    context: str | None = None
    priority: float = 0.0
    reasons: tuple[str, ...] = ()


@dataclass
class Verdict:
    trace_id: str
    score: float
    reason: str
    judge: str
    priority: float

    @property
    def passed(self) -> bool:
        return self.score >= 0.5


def select(
    candidates: list[Candidate],
    k: int,
    explore: float = 0.0,
    seed: int = 0,
) -> list[Candidate]:
    """The k highest-priority candidates, with round(k * explore) of them drawn at random instead."""
    if k <= 0 or not candidates:
        return []
    n_explore = min(round(k * explore), len(candidates))
    ranked = sorted(candidates, key=lambda c: -c.priority)
    chosen = [c for c in ranked if c.priority > 0][: k - n_explore]
    rest = [c for c in candidates if c not in chosen]
    rng = random.Random(seed)
    chosen += rng.sample(rest, min(k - len(chosen), len(rest)))
    return chosen[:k]


def escalate(
    candidates: list[Candidate],
    judge: Judge,
    budget: JudgeBudget | None = None,
    timeline=None,
    baseline_id: str = "escalation",
    component: str = "default",
    timestamp: str | None = None,
) -> list[Verdict]:
    """Judge candidates in order until the budget runs out; failed calls are skipped, not scored.

    With a TimelineStore, every verdict is stored as the "judge" series so the
    dashboard and attribution can line judge scores up with the signals.
    """
    granted = len(candidates) if budget is None else budget.try_spend(len(candidates))
    if granted < len(candidates):
        log.info("judge budget allows %d of %d calls", granted, len(candidates))
    verdicts = []
    for c in candidates[:granted]:
        try:
            score, reason = judge.judge(c.input, c.output, c.context)
        except (JudgeError, ImportError, OSError) as exc:
            log.warning("judge call for %s failed, not scored: %s", c.trace_id, exc)
            continue
        verdict = Verdict(c.trace_id, score, reason, judge.name, c.priority)
        verdicts.append(verdict)
        if timeline is not None:
            from datetime import datetime, timezone

            timeline.record(
                trace_id=c.trace_id,
                baseline_id=baseline_id,
                metric_name=JUDGE_METRIC,
                score=score,
                threshold=0.5,
                passed=verdict.passed,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
                metadata={"reason": reason, "judge": judge.name},
                component=component,
            )
    return verdicts
