import numpy as np
import pytest

from spaniq.statistical.psi import compute_psi


def test_identical_distributions_near_zero():
    data = np.random.default_rng(42).normal(0, 1, 1000)
    psi = compute_psi(data, data)
    assert psi < 0.01


def test_severe_shift_high_psi():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(5, 1, 1000)
    psi = compute_psi(baseline, current)
    assert psi > 0.25


def test_psi_non_negative():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(1, 1, 500)
    assert compute_psi(baseline, current) >= 0


def test_empty_baseline_raises():
    with pytest.raises(ValueError):
        compute_psi(np.array([]), np.array([1.0, 2.0]))


def test_empty_current_raises():
    with pytest.raises(ValueError):
        compute_psi(np.array([1.0, 2.0]), np.array([]))


def test_small_sample():
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    psi = compute_psi(baseline, current)
    assert psi >= 0


def test_deterministic():
    rng = np.random.default_rng(7)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(0.5, 1, 500)
    assert compute_psi(baseline, current) == compute_psi(baseline, current)
