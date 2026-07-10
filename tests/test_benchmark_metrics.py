"""Tests for the accuracy metrics module (Phase 4).

Positive class is "bad": a LOW score means the output looks bad, so
predict(score, threshold) flags 'bad' when score < threshold.
"""

from __future__ import annotations

import pytest

from benchmarks.analysis import metrics as m
from benchmarks.runners.spaniq_runner import LabeledResult, Prediction


def test_confusion_textbook():
    # bad outputs score low (0.2), good outputs score high (0.9); threshold 0.5
    labels = ["bad", "bad", "good", "good"]
    scores = [0.2, 0.2, 0.9, 0.9]
    c = m.confusion(labels, scores, threshold=0.5)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 0, 0, 2)
    assert m.precision(c) == pytest.approx(1.0)
    assert m.recall(c) == pytest.approx(1.0)
    assert m.f1(c) == pytest.approx(1.0)
    assert m.accuracy(c) == pytest.approx(1.0)


def test_confusion_with_errors():
    # one bad output slips through (scored high), one good output flagged (low)
    labels = ["bad", "bad", "good", "good"]
    scores = [0.2, 0.8, 0.3, 0.9]
    c = m.confusion(labels, scores, threshold=0.5)
    # bad@0.2 -> caught (tp); bad@0.8 -> missed (fn);
    # good@0.3 -> false alarm (fp); good@0.9 -> tn
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert m.precision(c) == pytest.approx(0.5)
    assert m.recall(c) == pytest.approx(0.5)
    assert m.f1(c) == pytest.approx(0.5)


def test_precision_recall_zero_denominators():
    # nothing flagged as bad -> precision 0/0 defined as 0.0
    c = m.confusion(["good", "good"], [0.9, 0.9], threshold=0.5)
    assert m.precision(c) == 0.0
    assert m.recall(c) == 0.0  # no positives either


def test_roc_auc_perfect_separation():
    labels = ["bad", "bad", "good", "good"]
    scores = [0.1, 0.2, 0.8, 0.9]  # all bad rank below all good
    assert m.roc_auc(labels, scores) == pytest.approx(1.0)


def test_roc_auc_reversed_is_zero():
    labels = ["bad", "bad", "good", "good"]
    scores = [0.9, 0.8, 0.2, 0.1]  # bad look better than good — worst case
    assert m.roc_auc(labels, scores) == pytest.approx(0.0)


def test_roc_auc_random_half():
    labels = ["bad", "good", "bad", "good"]
    scores = [0.5, 0.5, 0.5, 0.5]  # no separation, all ties
    assert m.roc_auc(labels, scores) == pytest.approx(0.5)


def test_average_precision_perfect():
    labels = ["bad", "bad", "good", "good"]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert m.average_precision(labels, scores) == pytest.approx(1.0)


def test_best_f1_threshold_separable():
    labels = ["bad", "bad", "good", "good"]
    scores = [0.2, 0.3, 0.7, 0.8]
    t = m.best_f1_threshold(labels, scores)
    # any threshold in (0.3, 0.7] perfectly separates
    c = m.confusion(labels, scores, t)
    assert m.f1(c) == pytest.approx(1.0)


def test_labeled_result_determinism_and_mean_scores():
    preds_a = [Prediction(0, "bad", 0.2), Prediction(1, "good", 0.9)]
    preds_b = [Prediction(0, "bad", 0.2), Prediction(1, "good", 0.9)]
    res = LabeledResult(tool="spaniq", dataset="qa", runs=[preds_a, preds_b])
    assert res.score_variance == pytest.approx(0.0)
    assert res.true_labels == ["bad", "good"]
    assert res.mean_scores == pytest.approx([0.2, 0.9])
