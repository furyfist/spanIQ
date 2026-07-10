"""Tests for PELT offline changepoint wrapper."""

import numpy as np
import pytest

from spaniq.attribution.changepoint.pelt import bic_penalty, detect_changepoints


def _series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    seg1 = rng.normal(0.7, 0.03, n // 2)
    seg2 = rng.normal(0.3, 0.03, n // 2)
    return np.concatenate([seg1, seg2])


def test_single_break_localized():
    series = _series()
    cps = detect_changepoints(series, penalty=3.0)
    assert len(cps) == 1
    assert abs(cps[0] - 100) <= 5


def test_two_breaks():
    rng = np.random.default_rng(0)
    s = np.concatenate(
        [rng.normal(0.7, 0.03, 100), rng.normal(0.3, 0.03, 100), rng.normal(0.7, 0.03, 100)]
    )
    cps = detect_changepoints(s, penalty=3.0)
    assert len(cps) == 2


def test_no_break_returns_empty():
    rng = np.random.default_rng(1)
    flat = rng.normal(0.7, 0.03, 200)
    cps = detect_changepoints(flat, penalty=3.0)
    assert cps == []


def test_min_size_respected():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.normal(0.7, 0.03, 5), rng.normal(0.3, 0.03, 5)])
    cps = detect_changepoints(s, penalty=3.0, min_size=10)
    assert cps == []


def test_deterministic():
    series = _series()
    assert detect_changepoints(series) == detect_changepoints(series)


def test_noise_robustness():
    rng = np.random.default_rng(99)
    flat = rng.normal(0.5, 0.5, 200)
    cps = detect_changepoints(flat, penalty=5.0)
    assert len(cps) <= 2


def test_bic_penalty():
    p = bic_penalty(100)
    assert p == pytest.approx(3.0 * np.log(100), rel=1e-3)
