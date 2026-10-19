"""Compare windows of per-output signals and decide whether a pipeline changed.

A single output says little; a shift in how a whole window of outputs is
distributed is what a model swap, prompt edit or broken retriever leaves
behind. compare() tests every signal with a KS test, corrects for testing many
signals at once (Benjamini-Hochberg), and only calls a signal drifted when the
shift is also large enough to matter (effect size or PSI), so a big window does
not turn every tiny wobble into an alarm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from spaniq.drift.stats import benjamini_hochberg, effect_size, mmd_test, psi
from spaniq.statistical.ks import compute_ks

DEFAULT_ALPHA = 0.01
DEFAULT_MIN_EFFECT = 0.5
DEFAULT_MIN_PSI = 0.2
MIN_SAMPLES = 5
_Z_CAP = 10.0
EMBEDDING = "embedding"


@dataclass
class SignalDrift:
    signal: str
    baseline_mean: float
    candidate_mean: float
    effect_size: float
    psi: float
    statistic: float
    p_value: float
    drifted: bool = False


@dataclass
class DriftReport:
    n_baseline: int
    n_candidate: int
    signals: list[SignalDrift] = field(default_factory=list)

    @property
    def drifted(self) -> bool:
        return any(s.drifted for s in self.signals)

    @property
    def drifted_signals(self) -> list[SignalDrift]:
        return [s for s in self.signals if s.drifted]

    def get(self, name: str) -> SignalDrift | None:
        return next((s for s in self.signals if s.signal == name), None)

    def to_dict(self) -> dict:
        return {
            "n_baseline": self.n_baseline,
            "n_candidate": self.n_candidate,
            "drifted": self.drifted,
            "signals": [
                {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}
                for d in (asdict(s) for s in self.signals)
            ],
        }


def _column(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.array([r[name] for r in rows if name in r], dtype=float)


def compare(
    baseline: list[dict[str, float]],
    candidate: list[dict[str, float]],
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = DEFAULT_MIN_EFFECT,
    min_psi: float = DEFAULT_MIN_PSI,
    baseline_embeddings: np.ndarray | None = None,
    candidate_embeddings: np.ndarray | None = None,
) -> DriftReport:
    """Test every signal present on both sides; drifted signals come first, largest shift first."""
    names = sorted({k for r in baseline for k in r} & {k for r in candidate for k in r})
    results: list[SignalDrift] = []
    for name in names:
        b, c = _column(baseline, name), _column(candidate, name)
        if len(b) < MIN_SAMPLES or len(c) < MIN_SAMPLES:
            continue
        if np.ptp(b) == 0 and np.ptp(c) == 0 and b[0] == c[0]:
            stat, p = 0.0, 1.0
        else:
            ks = compute_ks(b, c)
            stat, p = ks.statistic, ks.p_value
        results.append(
            SignalDrift(
                signal=name,
                baseline_mean=float(b.mean()),
                candidate_mean=float(c.mean()),
                effect_size=effect_size(b, c),
                psi=psi(b, c),
                statistic=stat,
                p_value=p,
            )
        )
    if (
        baseline_embeddings is not None
        and candidate_embeddings is not None
        and min(len(baseline_embeddings), len(candidate_embeddings)) >= MIN_SAMPLES
    ):
        stat, p = mmd_test(baseline_embeddings, candidate_embeddings)
        results.append(
            SignalDrift(
                signal=EMBEDDING,
                baseline_mean=float("nan"),
                candidate_mean=float("nan"),
                effect_size=float("nan"),
                psi=float("nan"),
                statistic=stat,
                p_value=p,
            )
        )

    for s, p_adj in zip(results, benjamini_hochberg([s.p_value for s in results]), strict=True):
        s.p_value = p_adj
        if s.signal == EMBEDDING:
            s.drifted = p_adj < alpha
        else:
            s.drifted = p_adj < alpha and (abs(s.effect_size) >= min_effect or s.psi >= min_psi)
    results.sort(key=lambda s: (not s.drifted, -abs(s.statistic)))
    return DriftReport(n_baseline=len(baseline), n_candidate=len(candidate), signals=results)


def drift_priorities(
    rows: list[dict[str, float]],
    baseline: list[dict[str, float]],
    report: DriftReport,
    prefix: str = "",
) -> list[tuple[float, list[str]]]:
    """For each row: summed z-score along every drifted signal's direction, and the top reasons."""
    directions = []
    for s in report.drifted_signals:
        values = [r[s.signal] for r in baseline if s.signal in r]
        if not values or s.effect_size != s.effect_size:
            continue
        std = float(np.std(values)) or max(abs(float(np.mean(values))) * 0.1, 0.05)
        directions.append((s.signal, float(np.mean(values)), std, float(np.sign(s.effect_size))))
    out = []
    for row in rows:
        scored = []
        for name, mean, std, sign in directions:
            if name not in row:
                continue
            z = min(_Z_CAP, max(0.0, sign * (row[name] - mean) / std))
            if z > 0:
                scored.append((z, f"{prefix}{name}={row[name]:.3g} (baseline mean {mean:.3g})"))
        scored.sort(reverse=True)
        out.append((sum(z for z, _ in scored), [r for _, r in scored[:3]]))
    return out


class WindowDriftDetector:
    """Online drift detection: a sliding window of recent signals against a fixed baseline.

    update() is called once per trace. Every `stride` traces, once the window
    holds at least `min_window` traces, the window is compared with the
    baseline. update() returns the report on the check where an incident starts
    and None otherwise; the incident ends at the first check that finds no drift.
    """

    def __init__(
        self,
        baseline: list[dict[str, float]],
        window: int = 100,
        stride: int = 25,
        min_window: int | None = None,
        alpha: float = DEFAULT_ALPHA,
        min_effect: float = DEFAULT_MIN_EFFECT,
        min_psi: float = DEFAULT_MIN_PSI,
    ) -> None:
        if len(baseline) < MIN_SAMPLES:
            raise ValueError(f"baseline needs at least {MIN_SAMPLES} traces, got {len(baseline)}")
        self.baseline = list(baseline)
        self.window: deque[dict[str, float]] = deque(maxlen=window)
        self.stride = stride
        self.min_window = min_window or max(MIN_SAMPLES, window // 2)
        self.alpha = alpha
        self.min_effect = min_effect
        self.min_psi = min_psi
        self.seen = 0
        self.in_incident = False
        self.last_report: DriftReport | None = None

    def update(self, signals: dict[str, float]) -> DriftReport | None:
        self.window.append(signals)
        self.seen += 1
        if len(self.window) < self.min_window or self.seen % self.stride != 0:
            return None
        report = compare(
            self.baseline, list(self.window), self.alpha, self.min_effect, self.min_psi
        )
        self.last_report = report
        if report.drifted and not self.in_incident:
            self.in_incident = True
            return report
        if not report.drifted:
            self.in_incident = False
        return None
