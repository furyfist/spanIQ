import math

import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.statistical.embeddings import embed, embed_reference
from spaniq.statistical.ks import compute_ks


def ks_critical_value(n: int, m: int, alpha: float) -> float:
    """Asymptotic one-sided two-sample KS critical value at level alpha."""
    return math.sqrt(-math.log(alpha) / 2) * math.sqrt((n + m) / (n * m))


class ConsistencyMetric(BaseMetric):
    """KS test on embedding distances: actual-to-baseline vs baseline-to-baseline.

    Detects when an output sits further from the baseline than baseline outputs
    sit from each other, i.e. the model became erratic. The test is one-sided,
    so an output that is closer than usual never fails. The score is the KS
    statistic (lower = more consistent).

    With threshold=None (the default) the pass line is the KS critical value at
    alpha for this baseline's sample sizes, recomputed on every measure and
    exposed as self.threshold. A fixed cut-off ignores sample size: with the 5
    to 30 baseline outputs typical here, random variation alone easily exceeds a
    small fixed value. Pass a float threshold to use a fixed cut-off instead.
    Requires >= 5 baseline_outputs.
    """

    def __init__(self, threshold: float | None = None, alpha: float = 0.05):
        super().__init__(threshold=threshold if threshold is not None else 0.0)
        self.alpha = alpha
        self._adaptive = threshold is None

    def measure(self, test_case: LLMTestCase) -> float:
        if len(test_case.baseline_outputs) < 5:
            raise ValueError("ConsistencyMetric requires at least 5 baseline_outputs")

        actual_vec = embed([test_case.actual_output])[0]
        baseline_vecs = embed_reference(test_case.baseline_outputs)

        actual_distances = 1 - baseline_vecs @ actual_vec
        upper = np.triu_indices(len(baseline_vecs), k=1)
        baseline_distances = 1 - (baseline_vecs @ baseline_vecs.T)[upper]

        ks_result = compute_ks(
            baseline_distances, actual_distances, alpha=self.alpha, alternative="greater"
        )
        if self._adaptive:
            self.threshold = ks_critical_value(
                len(actual_distances), len(baseline_distances), self.alpha
            )
        self.score = ks_result.statistic
        op = "<" if self.is_successful() else ">="
        self.reason = (
            f"KS statistic {self.score:.4f} {op} threshold {self.threshold:.4f} "
            f"(one-sided p={ks_result.p_value:.4f})"
        )
        return self.score

    def _check_threshold(self, score: float) -> bool:
        return score < self.threshold
