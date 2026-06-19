from __future__ import annotations

import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric


def _discrete_psi(baseline_freq: dict[str, int], current_freq: dict[str, int]) -> float:
    """PSI between two discrete token frequency distributions.

    Bypasses histogram binning — the distributions are already discrete.
    PSI < 0.1: no drift, 0.1-0.25: moderate, >0.25: significant.
    """
    all_tokens = list(set(baseline_freq) | set(current_freq))
    eps = 1e-6

    baseline_total = sum(baseline_freq.values()) + eps * len(all_tokens)
    current_total = sum(current_freq.values()) + eps * len(all_tokens)

    psi = 0.0
    for t in all_tokens:
        b = (baseline_freq.get(t, 0) + eps) / baseline_total
        c = (current_freq.get(t, 0) + eps) / current_total
        psi += (c - b) * np.log(c / b)
    return float(psi)


class ResponseDriftMetric(BaseMetric):
    """PSI on word-frequency distributions: baseline corpus vs actual_output.

    Stateless — no internal window. PipelineMonitor owns the rolling window
    and passes it as baseline_outputs. PSI < threshold passes (lower = less drift).
    """

    def __init__(self, threshold: float = 0.10):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.baseline_outputs:
            raise ValueError("ResponseDriftMetric requires baseline_outputs")

        baseline_dist = self._token_frequencies(" ".join(test_case.baseline_outputs))
        current_dist = self._token_frequencies(test_case.actual_output)

        self.score = _discrete_psi(baseline_dist, current_dist)
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
