from __future__ import annotations

from collections import deque

import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.statistical.psi import compute_psi


class ResponseDriftMetric(BaseMetric):
    """Rolling-window PSI on word-frequency distributions vs baseline corpus.

    Maintains a deque of the last window_size outputs and compares the
    aggregate word-frequency distribution against the baseline corpus.
    PSI < threshold passes (lower = less drift).
    """

    def __init__(self, threshold: float = 0.10, window_size: int = 20):
        super().__init__(threshold=threshold)
        self.window_size = window_size
        self._window: deque[str] = deque(maxlen=window_size)

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.baseline_outputs:
            raise ValueError("ResponseDriftMetric requires baseline_outputs")

        self._window.append(test_case.actual_output)

        if len(self._window) < 3:
            self.score = 0.0
            self.reason = f"warming up ({len(self._window)}/{self.window_size} traces)"
            return self.score

        baseline_dist = self._token_frequencies(" ".join(test_case.baseline_outputs))
        current_dist = self._token_frequencies(" ".join(self._window))

        all_tokens = list(set(baseline_dist) | set(current_dist))
        baseline_vec = np.array([baseline_dist.get(t, 0) for t in all_tokens], dtype=float)
        current_vec = np.array([current_dist.get(t, 0) for t in all_tokens], dtype=float)

        self.score = compute_psi(baseline_vec, current_vec)
        op = "<" if self.is_successful() else ">="
        self.reason = f"PSI {self.score:.4f} {op} threshold {self.threshold}"
        return self.score

    def _token_frequencies(self, text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for word in text.lower().split():
            counts[word] = counts.get(word, 0) + 1
        return counts

    def _check_threshold(self, score: float) -> bool:
        return score < self.threshold
