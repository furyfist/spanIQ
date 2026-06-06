from __future__ import annotations

from collections import deque

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
    """Rolling-window JS divergence on structural features vs baseline corpus.

    Features: char count, word count, sentence count, avg word length.
    Maintains a window of the last window_size outputs and compares the
    aggregate feature distribution against the baseline corpus.
    JS < threshold passes (lower = more stable).
    """

    def __init__(self, threshold: float = 0.15, window_size: int = 20):
        super().__init__(threshold=threshold)
        self.window_size = window_size
        self._window: deque[str] = deque(maxlen=window_size)

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.baseline_outputs:
            raise ValueError("OutputStabilityMetric requires baseline_outputs")

        self._window.append(test_case.actual_output)

        if len(self._window) < 3:
            self.score = 0.0
            self.reason = f"warming up ({len(self._window)}/{self.window_size} traces)"
            return self.score

        baseline = np.array([self._extract_features(o) for o in test_case.baseline_outputs])
        current = np.array([self._extract_features(o) for o in self._window])

        scores = []
        for i in range(baseline.shape[1]):
            baseline_col = baseline[:, i]
            current_col = current[:, i]
            if baseline_col.max() == baseline_col.min() and current_col.max() == current_col.min():
                scores.append(0.0 if baseline_col[0] == current_col[0] else 1.0)
            else:
                scores.append(_js_small_sample(baseline_col, current_col))

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
