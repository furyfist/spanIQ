import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.statistical.embeddings import cosine_similarity, embed
from spaniq.statistical.ks import compute_ks


class ConsistencyMetric(BaseMetric):
    """KS test on embedding distance distributions: actual-to-baseline vs baseline-to-baseline.

    Detects when a model becomes erratic — outputs vary more than historically normal.
    Requires >= 5 baseline_outputs for a meaningful distance distribution.
    KS statistic < threshold passes (lower = more consistent).
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        if len(test_case.baseline_outputs) < 5:
            raise ValueError("ConsistencyMetric requires at least 5 baseline_outputs")

        all_texts = [test_case.actual_output] + test_case.baseline_outputs
        vecs = embed(all_texts)
        actual_vec = vecs[0]
        baseline_vecs = vecs[1:]

        actual_distances = np.array([1 - cosine_similarity(actual_vec, bv) for bv in baseline_vecs])

        n = len(baseline_vecs)
        baseline_distances = np.array(
            [
                1 - cosine_similarity(baseline_vecs[i], baseline_vecs[j])
                for i in range(n)
                for j in range(i + 1, n)
            ]
        )

        ks_result = compute_ks(baseline_distances, actual_distances)
        self.score = ks_result.statistic
        op = "<" if self.is_successful() else ">="
        self.reason = (
            f"KS statistic {self.score:.4f} {op} threshold {self.threshold} "
            f"(p={ks_result.p_value:.4f})"
        )
        return self.score

    def _check_threshold(self, score: float) -> bool:
        return score < self.threshold
