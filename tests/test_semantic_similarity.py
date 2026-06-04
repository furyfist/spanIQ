import pytest

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric


def test_same_text_passes():
    tc = LLMTestCase(
        input="q",
        actual_output="we offer 30-day refunds",
        expected_output="we offer 30-day refunds",
    )
    metric = SemanticSimilarityMetric(threshold=0.7)
    metric.measure(tc)
    assert metric.is_successful()


def test_very_different_text_fails():
    tc = LLMTestCase(
        input="q",
        actual_output="quantum physics describes subatomic particles",
        expected_output="we offer 30-day refunds on all purchases",
    )
    metric = SemanticSimilarityMetric(threshold=0.7)
    metric.measure(tc)
    assert not metric.is_successful()


def test_uses_baseline_when_no_expected():
    tc = LLMTestCase(
        input="q",
        actual_output="refunds within 30 days",
        baseline_outputs=[
            "we offer 30-day refunds",
            "30-day return policy applies",
            "full refund available within 30 days",
        ],
    )
    metric = SemanticSimilarityMetric(threshold=0.5)
    score = metric.measure(tc)
    assert score >= 0


def test_no_reference_raises():
    tc = LLMTestCase(input="q", actual_output="some output")
    with pytest.raises(ValueError):
        SemanticSimilarityMetric().measure(tc)


def test_score_bounded():
    tc = LLMTestCase(
        input="q",
        actual_output="hello world",
        expected_output="hello world",
    )
    score = SemanticSimilarityMetric().measure(tc)
    assert -1.0 <= score <= 1.0


def test_deterministic():
    tc = LLMTestCase(
        input="q",
        actual_output="30-day money back guarantee",
        expected_output="we offer 30-day refunds",
    )
    m1, m2 = SemanticSimilarityMetric(), SemanticSimilarityMetric()
    assert m1.measure(tc) == m2.measure(tc)
