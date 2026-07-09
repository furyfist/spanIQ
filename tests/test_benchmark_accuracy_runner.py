"""Phase 5 — spanIQ runner emits labeled predictions for accuracy scoring."""
from __future__ import annotations

import pytest

from benchmarks.analysis import metrics as m
from benchmarks.config import DATASET_FILES
from benchmarks.runners.spaniq_runner import run_spaniq_predictions, LabeledResult, Prediction

QA_PATH = DATASET_FILES["qa_factual"]


def test_predictions_carry_labels():
    res = run_spaniq_predictions(QA_PATH, n_runs=2)
    assert isinstance(res, LabeledResult)
    assert res.tool == "spaniq"
    assert len(res.runs) == 2
    first = res.runs[0]
    assert all(isinstance(p, Prediction) for p in first)
    assert set(res.true_labels) == {"good", "bad"}


def test_predictions_are_deterministic():
    """spanIQ's determinism survives the prediction rewrite (secondary stat)."""
    res = run_spaniq_predictions(QA_PATH, n_runs=3)
    assert res.score_variance == pytest.approx(0.0, abs=1e-10)


def test_spaniq_separates_good_from_bad():
    """Sanity: on the labeled QA set, good answers should score higher than the
    wrong-entity bad answers, so AUC is clearly above chance."""
    res = run_spaniq_predictions(QA_PATH, n_runs=1)
    labels = res.true_labels
    scores = res.mean_scores
    auc = m.roc_auc(labels, scores)
    assert auc > 0.6, f"expected separation above chance, got AUC={auc:.3f}"
