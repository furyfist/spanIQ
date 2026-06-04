import pytest

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.output_stability import OutputStabilityMetric

BASELINE = [
    "we offer 30-day refunds on all purchases made through our store",
    "refunds are available within 30 days of your original purchase date",
    "you can return any item within 30 days for a complete refund",
    "our 30-day refund policy covers all standard purchases",
]


def make_case(actual: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual, baseline_outputs=BASELINE)


def test_score_is_float_in_valid_range():
    tc = make_case("refunds available within 30 days from the date of purchase")
    score = OutputStabilityMetric().measure(tc)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_extremely_short_output_detected():
    tc = make_case("no")
    metric = OutputStabilityMetric(threshold=0.15)
    score = metric.measure(tc)
    assert score > 0


def test_extremely_long_output_detected():
    tc = make_case("yes " * 200)
    metric = OutputStabilityMetric(threshold=0.15)
    score = metric.measure(tc)
    assert score > 0


def test_no_baseline_raises():
    tc = LLMTestCase(input="q", actual_output="a")
    with pytest.raises(ValueError, match="baseline_outputs"):
        OutputStabilityMetric().measure(tc)


def test_deterministic():
    tc = make_case("refunds processed within 30 days of purchase")
    m1, m2 = OutputStabilityMetric(), OutputStabilityMetric()
    assert m1.measure(tc) == m2.measure(tc)
