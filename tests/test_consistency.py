import pytest

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.consistency import ConsistencyMetric

BASELINE = [
    "we offer 30-day refunds on all purchases",
    "refunds are available within 30 days of purchase",
    "you can return any item within 30 days for a full refund",
    "our refund policy allows returns within 30 days",
    "30-day money back guarantee on all items",
]


def make_case(actual: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual, baseline_outputs=BASELINE)


def test_consistent_output_passes():
    tc = make_case("refunds available within 30 days from purchase")
    metric = ConsistencyMetric(threshold=0.6)
    metric.measure(tc)
    assert metric.is_successful()


def test_erratic_output_higher_ks_score():
    consistent_tc = make_case("30-day refund policy for all purchases")
    erratic_tc = make_case("quantum entanglement causes decoherence in neural networks")
    metric_c = ConsistencyMetric()
    metric_e = ConsistencyMetric()
    score_consistent = metric_c.measure(consistent_tc)
    score_erratic = metric_e.measure(erratic_tc)
    assert score_erratic >= score_consistent


def test_min_baseline_enforced():
    tc = LLMTestCase(
        input="q",
        actual_output="a",
        baseline_outputs=["one", "two", "three", "four"],
    )
    with pytest.raises(ValueError, match="5 baseline_outputs"):
        ConsistencyMetric().measure(tc)


def test_score_is_ks_statistic_bounded():
    tc = make_case("refunds processed within 30 days")
    score = ConsistencyMetric().measure(tc)
    assert 0.0 <= score <= 1.0


def test_deterministic():
    tc = make_case("30-day refund available for all customers")
    m1, m2 = ConsistencyMetric(), ConsistencyMetric()
    assert m1.measure(tc) == m2.measure(tc)
