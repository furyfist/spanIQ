import pytest

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.response_drift import ResponseDriftMetric

BASELINE = [
    "we offer 30-day refunds on all purchases",
    "refunds are available within 30 days of purchase",
    "you can return any item within 30 days for a full refund",
]


def make_case(actual: str) -> LLMTestCase:
    return LLMTestCase(input="refund policy?", actual_output=actual, baseline_outputs=BASELINE)


def test_identical_output_passes():
    tc = make_case("we offer 30-day refunds on all purchases")
    metric = ResponseDriftMetric(threshold=0.10)
    metric.measure(tc)
    assert metric.is_successful()


def test_completely_different_output_fails():
    tc = make_case("quantum entanglement causes superpositional decoherence in photons")
    metric = ResponseDriftMetric(threshold=0.10)
    metric.measure(tc)
    assert not metric.is_successful()


def test_no_baseline_raises():
    tc = LLMTestCase(input="q", actual_output="a")
    with pytest.raises(ValueError, match="baseline_outputs"):
        ResponseDriftMetric().measure(tc)


def test_threshold_respected():
    tc = make_case("we offer 30-day refunds on all purchases")
    strict = ResponseDriftMetric(threshold=0.001)
    strict.measure(tc)
    lenient = ResponseDriftMetric(threshold=10.0)
    lenient.measure(tc)
    assert lenient.is_successful()


def test_score_is_non_negative():
    tc = make_case("returns allowed within one month")
    metric = ResponseDriftMetric()
    score = metric.measure(tc)
    assert score >= 0


def test_deterministic():
    tc = make_case("30-day money back guarantee available")
    m1, m2 = ResponseDriftMetric(), ResponseDriftMetric()
    assert m1.measure(tc) == m2.measure(tc)
