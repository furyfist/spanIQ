import numpy as np
import pytest

from spaniq.statistical.js import compute_js


def test_identical_distributions_near_zero():
    data = np.random.default_rng(42).normal(0, 1, 500)
    assert compute_js(data, data) < 0.05


def test_severe_shift_high_js():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(10, 1, 500)
    assert compute_js(baseline, current) > 0.5


def test_bounded_zero_to_one():
    rng = np.random.default_rng(3)
    baseline = rng.normal(0, 1, 300)
    current = rng.normal(3, 2, 300)
    js = compute_js(baseline, current)
    assert 0.0 <= js <= 1.0


def test_symmetric():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 300)
    b = rng.normal(2, 1, 300)
    assert abs(compute_js(a, b) - compute_js(b, a)) < 1e-10


def test_empty_raises():
    with pytest.raises(ValueError):
        compute_js(np.array([]), np.array([1.0, 2.0]))


def test_deterministic():
    rng = np.random.default_rng(11)
    baseline = rng.normal(0, 1, 400)
    current = rng.normal(1, 1, 400)
    assert compute_js(baseline, current) == compute_js(baseline, current)
