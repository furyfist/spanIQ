"""Tests for CUSUM online changepoint primitive."""
import random
import pytest
from spaniq.attribution.changepoint.cusum import CusumState, cusum_update, run_cusum


def _stationary(n=300, mu=0.7, sigma=0.05, seed=0):
    random.seed(seed)
    return [random.gauss(mu, sigma) for _ in range(n)]


def _shifted(n=300, mu=0.7, sigma=0.05, break_at=100, shift=0.3, seed=0):
    random.seed(seed)
    s = [random.gauss(mu, sigma) for _ in range(n)]
    for i in range(break_at, n):
        s[i] = random.gauss(mu + shift, sigma)
    return s


def test_no_alarm_on_stationary():
    series = _stationary()
    state = run_cusum(series, mu0=0.7, k=0.025, h=5.0)
    assert state.alarm_index is None


def test_alarm_on_upward_shift():
    series = _shifted(shift=0.3)
    state = run_cusum(series, mu0=0.7, k=0.025, h=5.0)
    assert state.alarm_index is not None
    assert state.alarm_index > 80


def test_alarm_on_downward_shift():
    series = _shifted(shift=-0.3, mu=0.7)
    state = run_cusum(series, mu0=0.7, k=0.025, h=5.0)
    assert state.alarm_index is not None


def test_detection_delay_decreases_with_larger_shift():
    small = run_cusum(_shifted(shift=0.1), mu0=0.7, k=0.025, h=5.0)
    large = run_cusum(_shifted(shift=0.4), mu0=0.7, k=0.025, h=5.0)
    if small.alarm_index is not None and large.alarm_index is not None:
        assert large.alarm_index <= small.alarm_index


def test_stride_subsampling():
    series = _shifted(shift=0.4)
    s1 = run_cusum(series, mu0=0.7, k=0.025, h=5.0, stride=1)
    s2 = run_cusum(series, mu0=0.7, k=0.025, h=5.0, stride=5)
    assert s1.alarm_index is not None
    assert s2.alarm_index is not None


def test_deterministic():
    series = _shifted(shift=0.3)
    s1 = run_cusum(series, mu0=0.7, k=0.025, h=5.0)
    s2 = run_cusum(series, mu0=0.7, k=0.025, h=5.0)
    assert s1.alarm_index == s2.alarm_index


def test_cusum_update_increments_n_seen():
    state = CusumState()
    state = cusum_update(state, 0.7, mu0=0.7, k=0.025, h=5.0)
    assert state.n_seen == 1
    state = cusum_update(state, 0.7, mu0=0.7, k=0.025, h=5.0)
    assert state.n_seen == 2
