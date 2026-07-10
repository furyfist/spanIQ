"""Phase 5 — spanIQ runner emits labeled predictions for accuracy scoring."""

from __future__ import annotations

import pytest

from benchmarks.analysis import metrics as m
from benchmarks.config import DATASET_FILES
from benchmarks.runners.spaniq_runner import LabeledResult, Prediction, run_spaniq_predictions

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


def test_ragas_uses_the_v04_ascore_api():
    """Regression: the v0.4 collections metric exposes `ascore`, not
    `single_turn_ascore`. Calling the wrong name made every item fall back to a
    fabricated 0.5, which silently faked the ragas row."""
    ragas_collections = pytest.importorskip("ragas.metrics.collections")
    faithfulness = ragas_collections.Faithfulness
    assert hasattr(faithfulness, "ascore")
    assert not hasattr(faithfulness, "single_turn_ascore")


def test_ragas_raises_when_every_item_fails(monkeypatch):
    """A run where all items error is a broken integration, not data — it must
    raise instead of reporting a full column of 0.5 fallbacks."""
    pytest.importorskip("ragas.metrics.collections")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import benchmarks.runners.ragas_runner as rr

    class _AlwaysFails:
        async def ascore(self, **kwargs):
            raise RuntimeError("judge unreachable")

    monkeypatch.setattr(rr, "_get_ragas_llm", lambda: object())
    monkeypatch.setattr("ragas.metrics.collections.Faithfulness", lambda llm: _AlwaysFails())
    with pytest.raises(RuntimeError, match="every call failed"):
        rr.run_ragas_predictions(DATASET_FILES["rag_retrieval"], n_runs=1)


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
