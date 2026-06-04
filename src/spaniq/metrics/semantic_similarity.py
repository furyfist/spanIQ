import numpy as np

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.statistical.embeddings import cosine_similarity, embed


class SemanticSimilarityMetric(BaseMetric):
    """Cosine similarity between actual_output and expected_output or mean of baseline_outputs.

    Prefers expected_output if present, falls back to baseline mean.
    similarity >= threshold passes (higher = more similar).
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        if test_case.expected_output:
            vecs = embed([test_case.actual_output, test_case.expected_output])
            self.score = cosine_similarity(vecs[0], vecs[1])
        elif test_case.baseline_outputs:
            vecs = embed([test_case.actual_output] + test_case.baseline_outputs)
            actual_vec = vecs[0]
            baseline_mean = vecs[1:].mean(axis=0)
            baseline_mean = baseline_mean / np.linalg.norm(baseline_mean)
            self.score = cosine_similarity(actual_vec, baseline_mean)
        else:
            raise ValueError(
                "SemanticSimilarityMetric requires expected_output or baseline_outputs"
            )

        op = ">=" if self.is_successful() else "<"
        self.reason = f"similarity {self.score:.4f} {op} threshold {self.threshold}"
        return self.score

    def _check_threshold(self, score: float) -> bool:
        return score >= self.threshold
