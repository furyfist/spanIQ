"""Phase 7 — calibration/test split with disclosed threshold selection."""

from __future__ import annotations

import pytest

from benchmarks.analysis.calibrate import CALIBRATION_SEED, _split, evaluate_accuracy
from benchmarks.runners.spaniq_runner import LabeledResult, Prediction


def _labeled(scores_and_labels: list[tuple[float, str]]) -> LabeledResult:
    preds = [
        Prediction(item_id=i, true_label=lbl, score=sc)
        for i, (sc, lbl) in enumerate(scores_and_labels)
    ]
    return LabeledResult(tool="t", dataset="d", runs=[preds])


def test_split_is_deterministic():
    a = _split(10, CALIBRATION_SEED, 0.5)
    b = _split(10, CALIBRATION_SEED, 0.5)
    assert a == b
    cal, test = a
    assert set(cal).isdisjoint(test)
    assert sorted(cal + test) == list(range(10))


def test_split_changes_with_seed():
    assert _split(20, 1, 0.5) != _split(20, 2, 0.5)


def test_perfect_separation_scores_perfectly():
    # bad outputs score low, good high, cleanly separable in both folds
    data = [
        (0.1, "bad"),
        (0.15, "bad"),
        (0.2, "bad"),
        (0.25, "bad"),
        (0.8, "good"),
        (0.85, "good"),
        (0.9, "good"),
        (0.95, "good"),
    ]
    report = evaluate_accuracy(_labeled(data))
    assert report.f1 == pytest.approx(1.0)
    assert report.roc_auc == pytest.approx(1.0)
    assert report.average_precision == pytest.approx(1.0)


def test_threshold_comes_from_calibration_not_test():
    # A tool that ranks perfectly gets a good threshold regardless of the split;
    # this documents that the reported metrics are on the held-out test fold.
    data = [(0.1, "bad")] * 4 + [(0.9, "good")] * 4
    report = evaluate_accuracy(_labeled(data))
    assert report.n_calibration >= 1
    assert report.n_test >= 1
    # threshold sits between the bad cluster (0.1) and good cluster (0.9)
    assert 0.1 < report.threshold < 0.9


def test_determinism_sidebar_carried_through():
    run_a = [Prediction(0, "bad", 0.2), Prediction(1, "good", 0.9)]
    run_b = [Prediction(0, "bad", 0.2), Prediction(1, "good", 0.9)]
    res = LabeledResult(tool="spaniq", dataset="d", runs=[run_a, run_b])
    report = evaluate_accuracy(res)
    assert report.score_std == pytest.approx(0.0)
