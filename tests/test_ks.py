import numpy as np
import pytest

from spaniq.statistical.ks import compute_ks


def test_identical_distributions_not_significant():
    data = np.random.default_rng(42).normal(0, 1, 500)
    result = compute_ks(data, data)
    assert not result.significant


def test_shifted_distributions_significant():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(5, 1, 500)
    result = compute_ks(baseline, current)
    assert result.significant


def test_result_has_p_value():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 100)
    result = compute_ks(data, data)
    assert 0.0 <= result.p_value <= 1.0


def test_significant_is_python_bool():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 100)
    result = compute_ks(data, data)
    assert type(result.significant) is bool


def test_empty_raises():
    with pytest.raises(ValueError):
        compute_ks(np.array([]), np.array([1.0, 2.0]))


def test_deterministic():
    rng = np.random.default_rng(9)
    baseline = rng.normal(0, 1, 300)
    current = rng.normal(1, 1, 300)
    assert compute_ks(baseline, current) == compute_ks(baseline, current)
