"""Generate synthetic score series with known changepoints for validation."""

from __future__ import annotations

import numpy as np


def make_normal_series(
    n: int = 500,
    mu: float = 0.75,
    sigma: float = 0.04,
    ar_coef: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """AR(1) stationary series mimicking rolling-window autocorrelation."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma * np.sqrt(1 - ar_coef**2), n)
    series = np.empty(n)
    series[0] = mu
    for i in range(1, n):
        series[i] = mu + ar_coef * (series[i - 1] - mu) + noise[i]
    return np.clip(series, 0.0, 1.0)


def make_break_series(
    n: int = 500,
    mu_before: float = 0.75,
    mu_after: float | None = None,
    sigma: float = 0.04,
    ar_coef: float = 0.6,
    shift_sigma: float = 1.0,
    break_at: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Series with a single mean-shift changepoint at break_at.
    shift_sigma controls shift magnitude as multiples of sigma."""
    if break_at is None:
        break_at = n // 2
    if mu_after is None:
        mu_after = mu_before - shift_sigma * sigma
    rng = np.random.default_rng(seed)

    def _ar1(length, mu, start=None):
        arr = np.empty(length)
        arr[0] = mu if start is None else start
        noise = rng.normal(0, sigma * np.sqrt(1 - ar_coef**2), length)
        for i in range(1, length):
            arr[i] = mu + ar_coef * (arr[i - 1] - mu) + noise[i]
        return arr

    before = _ar1(break_at, mu_before)
    after = _ar1(n - break_at, mu_after, start=before[-1])
    return np.clip(np.concatenate([before, after]), 0.0, 1.0), break_at


def make_cascade_scenario(
    n: int = 300,
    root_break: int = 100,
    lead_gap: int = 5,
    sigma: float = 0.04,
    shift_sigma: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Two-component cascade: root breaks at root_break, follower breaks at root_break+lead_gap."""
    root_series, _ = make_break_series(
        n=n, break_at=root_break, sigma=sigma, shift_sigma=shift_sigma, seed=seed
    )
    follower_series, _ = make_break_series(
        n=n, break_at=root_break + lead_gap, sigma=sigma, shift_sigma=shift_sigma, seed=seed + 1
    )
    return root_series, follower_series, root_break, root_break + lead_gap
