from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric


def _js_small_sample(baseline_col: np.ndarray, current_col: np.ndarray) -> float:
    """JS distance between two small samples using adaptive bin count."""
    n = min(len(baseline_col), len(current_col))
    n_bins = max(3, min(10, n // 2))

    combined_min = min(baseline_col.min(), current_col.min())
    combined_max = max(baseline_col.max(), current_col.max())
    if combined_min == combined_max:
        return 0.0

    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)
    eps = 1e-6
    b_hist, _ = np.histogram(baseline_col, bins=bin_edges)
    c_hist, _ = np.histogram(current_col, bins=bin_edges)
    b_dist = (b_hist + eps) / (b_hist.sum() + eps * n_bins)
    c_dist = (c_hist + eps) / (c_hist.sum() + eps * n_bins)
    return float(jensenshannon(b_dist, c_dist))


class OutputStabilityMetric(BaseMetric):
    """JS divergence on structural features: baseline corpus vs actual_output.

    Stateless — no internal window. PipelineMonitor owns the rolling window
    and passes it as baseline_outputs. Features: char count, word count,
    sentence count, avg word length. JS < threshold passes (lower = more stable).
    """

    def __init__(self, threshold: float = 0.15):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.baseline_outputs:
            raise ValueError("OutputStabilityMetric requires baseline_outputs")

        baseline = np.array([self._extract_features(o) for o in test_case.baseline_outputs])
        current = self._extract_features(test_case.actual_output).reshape(1, -1)

        scores = []
        for i in range(baseline.shape[1]):
            baseline_col = baseline[:, i]
            current_val = current[0, i]
            if baseline_col.max() == baseline_col.min():
                scores.append(0.0 if baseline_col[0] == current_val else 1.0)
            else:
                scores.append(_js_small_sample(baseline_col, np.array([current_val])))

        self.score = float(np.mean(scores))
        op = "<" if self.is_successful() else ">="
        self.reason = f"JS stability {self.score:.4f} {op} threshold {self.threshold}"
        return self.score

    def _extract_features(self, text: str) -> np.ndarray:
        words = text.split()
        sentences = [s for s in text.split(".") if s.strip()]
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
        return np.array([len(text), len(words), len(sentences), avg_word_len])

    def _check_threshold(self, score: float) -> bool:
        return score < self.threshold
