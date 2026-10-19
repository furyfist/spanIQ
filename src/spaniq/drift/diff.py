"""Compare two runs of a pipeline: which units drifted, which component to blame, what to read first.

A run is a JSONL file of traces in the FileCollector format, for example the
same eval set replayed on main and on a pull request. Every unit (the final
output and each component span) is compared on its signals, and the candidate
traces that moved furthest in the direction of the drift are listed for review.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field

import numpy as np

from spaniq.drift.detector import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_EFFECT,
    DEFAULT_MIN_PSI,
    DriftReport,
    compare,
    drift_priorities,
)
from spaniq.monitor.trace import Trace
from spaniq.signals.extract import OUTPUT_UNIT, context_for, output_repeats_span, trace_signals
from spaniq.signals.nli import NLIModel

log = logging.getLogger(__name__)


@dataclass
class UnitDiff:
    unit: str
    position: float
    report: DriftReport


@dataclass
class ReviewItem:
    trace_id: str
    priority: float
    reasons: list[str]
    input: str
    output: str
    context: str | None = None


@dataclass
class DiffResult:
    n_baseline: int
    n_candidate: int
    units: list[UnitDiff]
    root_cause: str | None
    review: list[ReviewItem] = field(default_factory=list)
    verdicts: list = field(default_factory=list)

    @property
    def drifted(self) -> bool:
        return any(u.report.drifted for u in self.units)

    @property
    def drifted_units(self) -> list[str]:
        return [u.unit for u in self.units if u.report.drifted]

    def to_dict(self) -> dict:
        return {
            "n_baseline": self.n_baseline,
            "n_candidate": self.n_candidate,
            "drifted": self.drifted,
            "root_cause": self.root_cause,
            "units": [{"unit": u.unit, **u.report.to_dict()} for u in self.units],
            "review": [
                {
                    "trace_id": r.trace_id,
                    "priority": round(r.priority, 3),
                    "reasons": r.reasons,
                    "input": r.input,
                    "output": r.output,
                }
                for r in self.review
            ],
            "verdicts": [
                {
                    "trace_id": v.trace_id,
                    "score": round(v.score, 3),
                    "reason": v.reason,
                    "judge": v.judge,
                }
                for v in self.verdicts
            ],
        }


def load_traces(path: str) -> list[Trace]:
    """Traces from a JSONL file; malformed lines are skipped with a warning."""
    traces = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                traces.append(Trace.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                log.warning("%s:%d is not a trace with input and output, skipped", path, lineno)
    return traces


def _positions(traces: list[Trace]) -> dict[str, float]:
    """Median position of each component within its traces; the final output goes last."""
    seen: dict[str, list[int]] = {}
    for t in traces:
        for i, c in enumerate(t.components or []):
            seen.setdefault(c.name, []).append(i)
    positions = {name: float(statistics.median(idx)) for name, idx in seen.items()}
    positions[OUTPUT_UNIT] = float("inf")
    return positions


def upstream_root(drifted: list[str], positions: dict[str, float]) -> str | None:
    """The most upstream drifted component; the final output only when no component drifted."""
    if not drifted:
        return None
    return min(drifted, key=lambda u: positions.get(u, float("inf")))


def _embedding_unit(units: set[str], traces: list[Trace]) -> str | None:
    """The unit whose text is the final output: "output", or the span that produced it."""
    if OUTPUT_UNIT in units:
        return OUTPUT_UNIT
    for t in traces:
        for c in t.components or []:
            if c.output == t.output and c.name in units:
                return c.name
    return None


def _embed_outputs(traces: list[Trace]) -> np.ndarray:
    from spaniq.statistical.embeddings import embed

    return embed([t.output or " " for t in traces])


def _review(
    candidate: list[Trace],
    cand_signals: list[dict[str, dict[str, float]]],
    base_signals: dict[str, list[dict[str, float]]],
    units: list[UnitDiff],
    top_k: int,
) -> list[ReviewItem]:
    """Candidate traces ranked by how far they moved in the direction each drifted signal moved."""
    totals = [0.0] * len(candidate)
    reasons: list[list[tuple[float, str]]] = [[] for _ in candidate]
    for u in units:
        if not u.report.drifted:
            continue
        rows = [s.get(u.unit, {}) for s in cand_signals]
        ranked = drift_priorities(rows, base_signals.get(u.unit, []), u.report, f"{u.unit}.")
        for i, (priority, why) in enumerate(ranked):
            totals[i] += priority
            reasons[i] += [(priority, r) for r in why]
    items = [
        ReviewItem(
            trace_id=t.trace_id,
            priority=totals[i],
            reasons=[r for _, r in sorted(reasons[i], key=lambda x: -x[0])[:3]],
            input=t.input,
            output=t.output,
            context=context_for(t),
        )
        for i, t in enumerate(candidate)
        if totals[i] > 0
    ]
    items.sort(key=lambda r: -r.priority)
    return items[:top_k]


def diff_traces(
    baseline: list[Trace],
    candidate: list[Trace],
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = DEFAULT_MIN_EFFECT,
    min_psi: float = DEFAULT_MIN_PSI,
    embeddings: bool = False,
    nli: NLIModel | None = None,
    top_k: int = 10,
) -> DiffResult:
    """Compare every unit's signals between two runs. embeddings adds an MMD test on final outputs."""
    base_rows = [trace_signals(t, nli) for t in baseline]
    cand_rows = [trace_signals(t, nli) for t in candidate]
    base_by_unit: dict[str, list[dict[str, float]]] = {}
    cand_by_unit: dict[str, list[dict[str, float]]] = {}
    for rows, by_unit in ((base_rows, base_by_unit), (cand_rows, cand_by_unit)):
        for units in rows:
            for unit, signals in units.items():
                by_unit.setdefault(unit, []).append(signals)

    positions = _positions(baseline + candidate)
    shared = set(base_by_unit) & set(cand_by_unit)
    if len(shared) > 1 and output_repeats_span(baseline + candidate):
        # the final output repeats a component's output; reporting it twice adds noise
        shared.discard(OUTPUT_UNIT)
    results = []
    for unit in sorted(shared, key=lambda u: positions.get(u, 0)):
        emb_b = emb_c = None
        if embeddings and unit == _embedding_unit(shared, baseline):
            emb_b, emb_c = _embed_outputs(baseline), _embed_outputs(candidate)
        report = compare(
            base_by_unit[unit],
            cand_by_unit[unit],
            alpha,
            min_effect,
            min_psi,
            baseline_embeddings=emb_b,
            candidate_embeddings=emb_c,
        )
        results.append(UnitDiff(unit=unit, position=positions.get(unit, 0.0), report=report))

    drifted = [u.unit for u in results if u.report.drifted]
    return DiffResult(
        n_baseline=len(baseline),
        n_candidate=len(candidate),
        units=results,
        root_cause=upstream_root(drifted, positions),
        review=_review(candidate, cand_rows, base_by_unit, results, top_k),
    )
