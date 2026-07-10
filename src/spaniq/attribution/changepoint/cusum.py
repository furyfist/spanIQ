from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CusumState:
    s_pos: float = 0.0
    s_neg: float = 0.0
    alarm_index: int | None = None
    n_seen: int = 0


def cusum_update(
    state: CusumState,
    x: float,
    mu0: float,
    k: float,
    h: float,
) -> CusumState:
    """One online CUSUM step (Page 1954). Two-sided: detects upward shifts
    (drift scores rising) and downward (similarity scores falling).
    k = slack (typically 0.5 * baseline sigma), h = decision threshold.
    Returns new state; state.alarm_index is set on the first crossing only."""
    s_pos = max(0.0, state.s_pos + (x - mu0 - k))
    s_neg = max(0.0, state.s_neg + (mu0 - x - k))
    n_seen = state.n_seen + 1
    alarm_index = state.alarm_index
    if alarm_index is None and (s_pos > h or s_neg > h):
        alarm_index = n_seen - 1
    return CusumState(s_pos=s_pos, s_neg=s_neg, alarm_index=alarm_index, n_seen=n_seen)


def cusum_reset(state: CusumState) -> CusumState:
    """Reset accumulators after an alarm while preserving n_seen."""
    return CusumState(s_pos=0.0, s_neg=0.0, alarm_index=None, n_seen=state.n_seen)


def run_cusum(
    series: list[float],
    mu0: float,
    k: float,
    h: float,
    stride: int = 1,
) -> CusumState:
    """Run CUSUM over a full series with optional stride subsampling.
    stride > 1 reduces autocorrelation from rolling-window scores."""
    state = CusumState()
    for i, x in enumerate(series):
        if stride > 1 and i % stride != 0:
            continue
        state = cusum_update(state, x, mu0, k, h)
        if state.alarm_index is not None:
            break
    return state
