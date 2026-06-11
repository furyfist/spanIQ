from __future__ import annotations

import numpy as np

from spaniq.attribution.changepoint.cusum import CusumState, cusum_update


def _mean_run_length(
    series: np.ndarray,
    mu0: float,
    k: float,
    h: float,
    stride: int = 1,
) -> float:
    """Compute mean run length between false alarms on a no-break series."""
    runs: list[int] = []
    state = CusumState()
    run_start = 0
    for i, x in enumerate(series):
        if stride > 1 and i % stride != 0:
            continue
        state = cusum_update(state, float(x), mu0, k, h)
        if state.alarm_index is not None:
            runs.append(state.n_seen - run_start)
            run_start = state.n_seen
            state = CusumState(n_seen=state.n_seen)
    if not runs:
        return float(len(series))
    return float(np.mean(runs))


def calibrate_h(
    normal_series: list[np.ndarray],
    k: float,
    mu0: float | None = None,
    target_arl0: int = 500,
    stride: int = 1,
    h_candidates: list[float] | None = None,
) -> float:
    """Pick smallest h whose mean run length between false alarms >= target_arl0.
    Evaluates over concatenated normal_series at each candidate h value.
    mu0 defaults to mean of the concatenated series if not provided."""
    combined = np.concatenate(normal_series)
    if mu0 is None:
        mu0 = float(np.mean(combined))
    if h_candidates is None:
        h_candidates = [float(v) for v in np.arange(0.5, 20.1, 0.5)]
    for h in sorted(h_candidates):
        arl = _mean_run_length(combined, mu0=mu0, k=k, h=h, stride=stride)
        if arl >= target_arl0:
            return h
    return h_candidates[-1]
