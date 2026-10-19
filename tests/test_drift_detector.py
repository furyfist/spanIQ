import numpy as np
import pytest

from spaniq.drift import (
    EMBEDDING,
    WindowDriftDetector,
    benjamini_hochberg,
    compare,
    effect_size,
    mmd_test,
    psi,
)


def _rows(rng, n, mean=10.0, flag_rate=0.0):
    return [
        {"chars": float(rng.normal(mean, 1.0)), "refusal": float(rng.random() < flag_rate)}
        for _ in range(n)
    ]


def test_effect_size_sign_and_cap():
    assert effect_size(np.array([0.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(2.0)
    assert effect_size(np.zeros(5), np.zeros(5)) == 0.0
    assert effect_size(np.zeros(5), np.ones(5)) == 99.0


def test_psi_handles_flags():
    assert psi(np.zeros(50), np.zeros(50)) == pytest.approx(0.0, abs=1e-9)
    assert psi(np.zeros(50), np.r_[np.zeros(25), np.ones(25)]) > 1.0


def test_benjamini_hochberg():
    assert benjamini_hochberg([]) == []
    adj = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03, 0.5], strict=True))


def test_same_distribution_does_not_drift():
    rng = np.random.default_rng(0)
    report = compare(_rows(rng, 200), _rows(rng, 200))
    assert not report.drifted


def test_shift_drifts_and_is_listed_first():
    rng = np.random.default_rng(1)
    report = compare(_rows(rng, 200), _rows(rng, 100, mean=12.0, flag_rate=0.3))
    assert report.drifted
    assert {s.signal for s in report.drifted_signals} == {"chars", "refusal"}
    assert report.signals[0].drifted
    assert report.get("chars").effect_size > 1.5


def test_tiny_shift_on_big_sample_is_not_called_drift():
    """Significant but negligible: the effect-size gate keeps it quiet."""
    rng = np.random.default_rng(2)
    report = compare(_rows(rng, 5000), _rows(rng, 5000, mean=10.15))
    chars = report.get("chars")
    assert chars.p_value < 0.01
    assert not chars.drifted


def test_constant_equal_signal_is_skipped_gracefully():
    report = compare([{"x": 1.0}] * 10, [{"x": 1.0}] * 10)
    assert report.get("x").p_value == 1.0
    assert not report.drifted


def test_too_few_samples_are_ignored():
    assert compare([{"x": 1.0}] * 3, [{"x": 5.0}] * 3).signals == []


def test_signals_missing_on_one_side_are_ignored():
    report = compare([{"a": 1.0, "b": 2.0}] * 10, [{"a": 1.0}] * 10)
    assert [s.signal for s in report.signals] == ["a"]


def test_embedding_mmd():
    rng = np.random.default_rng(3)
    same, p_same = mmd_test(rng.normal(size=(60, 8)), rng.normal(size=(60, 8)), n_permutations=100)
    moved, p_moved = mmd_test(
        rng.normal(size=(60, 8)), rng.normal(1.0, size=(60, 8)), n_permutations=100
    )
    assert p_moved < 0.05 < p_same
    assert moved > same
    report = compare(
        [{"x": 1.0}] * 60,
        [{"x": 1.0}] * 60,
        baseline_embeddings=rng.normal(size=(60, 8)),
        candidate_embeddings=rng.normal(1.0, size=(60, 8)),
    )
    assert report.get(EMBEDDING).drifted


def test_to_dict_has_no_nan():
    rng = np.random.default_rng(4)
    d = compare(
        [{"x": 1.0}] * 10,
        [{"x": 1.0}] * 10,
        baseline_embeddings=rng.normal(size=(10, 4)),
        candidate_embeddings=rng.normal(size=(10, 4)),
    ).to_dict()
    emb = next(s for s in d["signals"] if s["signal"] == EMBEDDING)
    assert emb["effect_size"] is None


def test_window_detector_one_alarm_per_incident():
    rng = np.random.default_rng(5)
    det = WindowDriftDetector(_rows(rng, 200), window=60, stride=10)
    alarms = [i for i, r in enumerate(_rows(rng, 300)) if det.update(r)]
    assert alarms == []
    shifted = _rows(rng, 300, mean=14.0)
    alarms = [i for i, r in enumerate(shifted) if det.update(r)]
    assert len(alarms) == 1
    assert det.in_incident
    assert det.last_report.drifted
    recovered = _rows(rng, 200)
    for r in recovered:
        det.update(r)
    assert not det.in_incident


def test_window_detector_needs_baseline():
    with pytest.raises(ValueError, match="at least"):
        WindowDriftDetector([{"x": 1.0}])
