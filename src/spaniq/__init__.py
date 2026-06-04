from spaniq.core.eval_result import CostReport, EvalResult, MetricResult, TestCaseResult
from spaniq.core.evaluate import evaluate
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.metrics.consistency import ConsistencyMetric
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric


def assert_eval(test_case: LLMTestCase, metrics: list[BaseMetric]) -> None:
    """Pytest-friendly assertion. Raises AssertionError if any metric fails."""
    result = evaluate([test_case], metrics, verbose=False)
    tc_result = result.test_case_results[0]
    if not tc_result.passed:
        failures = [
            f"{mr.metric_name}: {mr.score:.4f} (threshold: {mr.threshold})"
            for mr in tc_result.metric_results
            if not mr.passed
        ]
        raise AssertionError(f"Metrics failed: {', '.join(failures)}")


__all__ = [
    "evaluate",
    "assert_eval",
    "LLMTestCase",
    "EvalResult",
    "MetricResult",
    "TestCaseResult",
    "CostReport",
    "ResponseDriftMetric",
    "SemanticSimilarityMetric",
    "OutputStabilityMetric",
    "ConsistencyMetric",
]
