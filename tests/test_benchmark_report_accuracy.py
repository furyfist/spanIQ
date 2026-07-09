"""Phase 8 — accuracy-first reporting with per-prediction audit trail."""
from __future__ import annotations

import csv
import pathlib
import tempfile

from benchmarks.analysis.report_accuracy import (
    build_reports, save_accuracy_csv, save_predictions_csv, save_summary_md,
)
from benchmarks.runners.spaniq_runner import LabeledResult, Prediction


def _result(tool: str) -> LabeledResult:
    preds = [
        Prediction(0, "bad", 0.1, "wrong_entity"),
        Prediction(1, "bad", 0.2, "hallucination"),
        Prediction(2, "good", 0.85, None),
        Prediction(3, "good", 0.95, None),
    ]
    return LabeledResult(tool=tool, dataset="qa_factual", runs=[preds, preds])


def test_accuracy_csv_has_pr_columns():
    reports = build_reports([_result("spaniq")])
    with tempfile.TemporaryDirectory() as tmp:
        path = save_accuracy_csv(reports, pathlib.Path(tmp))
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert rows[0]["tool"] == "spaniq"
        for col in ("precision", "recall", "f1", "roc_auc", "avg_precision"):
            assert col in rows[0]
        assert float(rows[0]["f1"]) == 1.0  # cleanly separable


def test_predictions_csv_is_recomputable():
    """The audit trail carries every item's label + score so P/R can be rebuilt."""
    with tempfile.TemporaryDirectory() as tmp:
        path = save_predictions_csv([_result("spaniq")], pathlib.Path(tmp))
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert len(rows) == 4
        assert {r["true_label"] for r in rows} == {"bad", "good"}
        assert rows[0]["failure_kind"] == "wrong_entity"


def test_summary_md_mentions_recall_and_determinism():
    reports = build_reports([_result("spaniq")])
    with tempfile.TemporaryDirectory() as tmp:
        path = save_summary_md(reports, pathlib.Path(tmp))
        text = path.read_text(encoding="utf-8").lower()
        assert "recall" in text
        assert "determinism" in text
        assert "positive class" in text
