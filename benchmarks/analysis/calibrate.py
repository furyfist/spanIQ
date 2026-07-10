"""Threshold calibration — the fairness rule of the accuracy benchmark.

Applied identically to every tool (see docs/plans/benchmark_v2_accuracy.md §
"Threshold selection"):

  1. Split items into a calibration fold and a test fold with a fixed seed.
  2. Pick the F1-maximizing threshold on the calibration fold only.
  3. Report precision / recall / F1 on the test fold using that threshold.
  4. Also report AUC / average precision on the test fold (threshold-free).

No tool — including spanIQ — is scored with a threshold tuned on the data it is
graded on. The seed is fixed and committed, so the method is reproducible; only
which items land in which fold depends on the seed, never the procedure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from benchmarks.analysis import metrics as m
from benchmarks.runners.spaniq_runner import LabeledResult

CALIBRATION_SEED = 20260710  # fixed & committed for reproducibility
CALIBRATION_FRACTION = 0.5


@dataclass
class AccuracyReport:
    tool: str
    dataset: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    n_calibration: int
    n_test: int
    # determinism sidebar, carried through from the LabeledResult
    score_std: float

    def as_row(self) -> dict:
        return {
            "tool": self.tool,
            "dataset": self.dataset,
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "avg_precision": round(self.average_precision, 4),
            "n_calibration": self.n_calibration,
            "n_test": self.n_test,
            "score_std": round(self.score_std, 4),
        }


def _split(n: int, seed: int, frac: float) -> tuple[list[int], list[int]]:
    """Deterministic index split into (calibration, test)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(round(n * frac)))
    return sorted(idx[:cut]), sorted(idx[cut:])


def evaluate_accuracy(
    result: LabeledResult,
    seed: int = CALIBRATION_SEED,
    frac: float = CALIBRATION_FRACTION,
) -> AccuracyReport:
    """Turn a tool's predictions into a test-fold accuracy report.

    Uses the per-item mean score across runs so a tool that is slightly
    non-deterministic is judged on its central tendency, not one lucky run.
    """
    labels = result.true_labels
    scores = result.mean_scores
    n = len(labels)

    cal_idx, test_idx = _split(n, seed, frac)
    # fall back to using all data for both folds if a class is missing from one
    cal_labels = [labels[i] for i in cal_idx]
    if len(set(cal_labels)) < 2 or not test_idx:
        cal_idx = test_idx = list(range(n))

    cal_labels = [labels[i] for i in cal_idx]
    cal_scores = [scores[i] for i in cal_idx]
    test_labels = [labels[i] for i in test_idx]
    test_scores = [scores[i] for i in test_idx]

    threshold = m.best_f1_threshold(cal_labels, cal_scores)
    conf = m.confusion(test_labels, test_scores, threshold)

    return AccuracyReport(
        tool=result.tool,
        dataset=result.dataset,
        threshold=threshold,
        precision=m.precision(conf),
        recall=m.recall(conf),
        f1=m.f1(conf),
        roc_auc=m.roc_auc(test_labels, test_scores),
        average_precision=m.average_precision(test_labels, test_scores),
        n_calibration=len(cal_idx),
        n_test=len(test_idx),
        score_std=result.score_std,
    )
