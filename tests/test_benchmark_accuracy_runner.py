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


def test_predictions_from_scores_attaches_dataset_labels():
    """The label always comes from the dataset row, never from the tool."""
    from benchmarks.runners.spaniq_runner import predictions_from_scores
    rows = [
        {"label": "bad", "failure_kind": "wrong_entity"},
        {"label": "good", "failure_kind": None},
    ]
    preds = predictions_from_scores(rows, [0.1, 0.95])
    assert [p.true_label for p in preds] == ["bad", "good"]
    assert preds[0].failure_kind == "wrong_entity"
    assert [p.score for p in preds] == [0.1, 0.95]


def test_competitor_prediction_runners_skip_without_key(monkeypatch):
    """Every competitor's prediction path skips cleanly with no Groq key."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from benchmarks.runners.groq_runner import run_groq_predictions
    from benchmarks.runners.langfuse_runner import run_langfuse_predictions
    from benchmarks.runners.ragas_runner import run_ragas_predictions

    rag_path = DATASET_FILES["rag_retrieval"]
    with pytest.raises((ImportError, EnvironmentError)):
        run_groq_predictions(QA_PATH, n_runs=1)
    with pytest.raises((ImportError, EnvironmentError)):
        run_langfuse_predictions(QA_PATH, n_runs=1)
    with pytest.raises((ImportError, EnvironmentError, ValueError)):
        run_ragas_predictions(rag_path, n_runs=1)
