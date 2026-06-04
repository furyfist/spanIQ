import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.statistical.js import compute_js


class OutputStabilityMetric(BaseMetric):
    """JS divergence on structural features between actual_output and baseline_outputs.

    Features: char count, word count, sentence count, avg word length.
    Detects length/structure changes without semantic analysis.
    JS < threshold passes (lower = more stable).
    """

    def __init__(self, threshold: float = 0.15):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.baseline_outputs:
            raise ValueError("OutputStabilityMetric requires baseline_outputs")

        current = self._extract_features(test_case.actual_output)
        baseline = np.array([self._extract_features(o) for o in test_case.baseline_outputs])

        scores = []
        for i in range(current.shape[0]):
            baseline_col = baseline[:, i]
            current_arr = np.array([current[i]])
            if baseline_col.max() == baseline_col.min():
                scores.append(0.0 if current[i] == baseline_col[0] else 1.0)
            else:
                scores.append(compute_js(baseline_col, current_arr))

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
