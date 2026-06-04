import pytest

from spaniq import assert_eval
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

BASELINE = [
    "we offer 30-day refunds on all purchases",
    "refunds are available within 30 days of purchase",
    "you can return any item within 30 days for a full refund",
]


def test_assert_eval_passes_for_good_output():
    tc = LLMTestCase(
        input="refund policy?",
        actual_output="we offer 30-day refunds on all purchases",
        baseline_outputs=BASELINE,
    )
    assert_eval(tc, [ResponseDriftMetric(threshold=10.0)])


def test_assert_eval_raises_for_bad_output():
    tc = LLMTestCase(
        input="refund policy?",
        actual_output="quantum entanglement superposition decoherence photons",
        expected_output="we offer 30-day refunds",
    )
    with pytest.raises(AssertionError, match="Metrics failed"):
        assert_eval(tc, [SemanticSimilarityMetric(threshold=0.99)])


def test_assert_eval_message_contains_metric_name():
    tc = LLMTestCase(
        input="q",
        actual_output="completely unrelated output about rockets",
        expected_output="30-day refund policy",
    )
    with pytest.raises(AssertionError) as exc_info:
        assert_eval(tc, [SemanticSimilarityMetric(threshold=0.99)])
    assert "SemanticSimilarityMetric" in str(exc_info.value)


def test_public_imports():
    from spaniq import (
        ConsistencyMetric,
        CostReport,
        EvalResult,
        LLMTestCase,
        MetricResult,
        OutputStabilityMetric,
        ResponseDriftMetric,
        SemanticSimilarityMetric,
        TestCaseResult,
        assert_eval,
        evaluate,
    )

    assert all(
        x is not None
        for x in [
            evaluate,
            assert_eval,
            LLMTestCase,
            EvalResult,
            MetricResult,
            TestCaseResult,
            CostReport,
            ResponseDriftMetric,
            SemanticSimilarityMetric,
            OutputStabilityMetric,
            ConsistencyMetric,
        ]
    )
