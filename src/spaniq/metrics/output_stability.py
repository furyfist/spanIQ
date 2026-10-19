from __future__ import annotations

import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric

FEATURES = ("char_count", "word_count", "sentence_count", "avg_word_length")


def _range_deviation(baseline_col: np.ndarray, value: float) -> float:
    """How far value falls outside the baseline's observed range, in [0, 1].

    0 inside the range; outside it grows with the distance, scaled by the range
    width (at least a quarter of the median, and at least 1) and capped at 1.
    The floor keeps a constant baseline feature from turning any change into a
    hard 1.0.
    """
    lo, hi = float(baseline_col.min()), float(baseline_col.max())
    beyond = max(0.0, lo - value, value - hi)
    if beyond == 0.0:
        return 0.0
    scale = max(hi - lo, 0.25 * abs(float(np.median(baseline_col))), 1.0)
    return min(1.0, beyond / scale)


class OutputStabilityMetric(BaseMetric):
    """Structural deviation of actual_output from the baseline outputs.

    Features: char count, word count, sentence count, average word length. Each
    feature scores 0 while it stays inside the range the baseline outputs span
    and rises toward 1 the further outside it lands; the score is the mean over
    features. Stateless: callers choose what to pass as baseline_outputs.
    score < threshold passes (lower = more stable).

    An earlier version computed a JS divergence between the baseline histogram
    and the single current value, which was close to all-or-nothing.
    """

    def __init__(self, threshold: float = 0.15):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.baseline_outputs:
            raise ValueError("OutputStabilityMetric requires baseline_outputs")

        baseline = np.array([self._extract_features(o) for o in test_case.baseline_outputs])
        current = self._extract_features(test_case.actual_output)

        scores = [_range_deviation(baseline[:, i], current[i]) for i in range(len(FEATURES))]
        self.score = float(np.mean(scores))
        worst = FEATURES[int(np.argmax(scores))]
        op = "<" if self.is_successful() else ">="
        self.reason = (
            f"structural deviation {self.score:.4f} {op} threshold {self.threshold} "
            f"(largest: {worst})"
        )
        return self.score

    def _extract_features(self, text: str) -> np.ndarray:
        words = text.split()
        sentences = [s for s in text.split(".") if s.strip()]
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
        return np.array([len(text), len(words), len(sentences), avg_word_len], dtype=float)

    def _check_threshold(self, score: float) -> bool:
        return score < self.threshold
