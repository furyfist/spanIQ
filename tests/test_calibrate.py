"""Tests for empirical CUSUM h calibration."""

import numpy as np

from spaniq.attribution.changepoint.calibrate import calibrate_h


def _normal_series(seed=0, n=2000, mu=0.7, sigma=0.05):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


def test_calibrate_returns_positive_h():
    series = [_normal_series(i) for i in range(3)]
    h = calibrate_h(series, k=0.025, mu0=0.7, target_arl0=100)
    assert h > 0


def test_higher_arl_gives_higher_h():
    series = [_normal_series(i) for i in range(3)]
    h_low = calibrate_h(series, k=0.025, mu0=0.7, target_arl0=50)
    h_high = calibrate_h(series, k=0.025, mu0=0.7, target_arl0=500)
    assert h_high >= h_low


def test_calibrate_uses_mu0_when_provided():
    series = [_normal_series(i) for i in range(2)]
    h = calibrate_h(series, k=0.025, mu0=0.7, target_arl0=50)
    assert h > 0
