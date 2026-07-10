"""E2E test: the accuracy benchmark runs end to end on labeled data.

The primary assertion is accuracy — that spanIQ separates good outputs from bad
ones on a labeled dataset. Determinism and $0 cost are still asserted, but as
the secondary facts they are: they hold by construction (no LLM call), and they
are not evidence that spanIQ is better than an LLM judge.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from benchmarks.analysis import metrics as m
from benchmarks.analysis.calibrate import (
    CALIBRATION_FRACTION, CALIBRATION_SEED, _split, evaluate_accuracy,
)
from benchmarks.analysis.report_accuracy import build_reports, save_predictions_csv
from benchmarks.config import DATASET_FILES
from benchmarks.runners.spaniq_runner import run_spaniq_eval, run_spaniq_predictions

QA_PATH = DATASET_FILES["qa_factual"]


def test_e2e_accuracy_separates_good_from_bad():
    """The benchmark's real claim: the tool detects bad outputs above chance."""
    result = run_spaniq_predictions(QA_PATH, n_runs=2)
    report = evaluate_accuracy(result)

    assert report.tool == "spaniq"
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert report.roc_auc > 0.5, (
        f"spanIQ should rank bad outputs below good ones, got AUC={report.roc_auc:.3f}"
    )
    assert report.f1 > 0.5, f"expected usable F1, got {report.f1:.3f}"


def test_e2e_accuracy_writes_recomputable_audit_trail():
    """Every reported number must be rebuildable from the raw predictions."""
    result = run_spaniq_predictions(QA_PATH, n_runs=1)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_predictions_csv([result], pathlib.Path(tmp))
        text = path.read_text(encoding="utf-8")
        assert "true_label" in text and "mean_score" in text

    # Recompute the report's metrics from the same held-out fold it used.
    report = build_reports([result])[0]
    cal_idx, test_idx = _split(len(result.true_labels), CALIBRATION_SEED, CALIBRATION_FRACTION)
    labels = [result.true_labels[i] for i in test_idx]
    scores = [result.mean_scores[i] for i in test_idx]

    conf = m.confusion(labels, scores, report.threshold)
    assert m.precision(conf) == pytest.approx(report.precision, abs=1e-9)
    assert m.recall(conf) == pytest.approx(report.recall, abs=1e-9)
    assert m.f1(conf) == pytest.approx(report.f1, abs=1e-9)


def test_e2e_benchmark_determinism_and_cost_are_by_construction():
    """Secondary facts: spanIQ makes no LLM call, so it is free and repeatable."""
    result = run_spaniq_eval(QA_PATH, n_runs=3)
    assert result.score_variance == pytest.approx(0.0, abs=1e-10)
    assert result.total_cost_usd == 0.0
