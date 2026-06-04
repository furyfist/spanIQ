from spaniq.core.evaluate import evaluate
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.response_drift import ResponseDriftMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

BASELINE = [
    "we offer 30-day refunds on all purchases",
    "refunds are available within 30 days of purchase",
    "you can return any item within 30 days for a full refund",
]


def make_case(actual: str) -> LLMTestCase:
    return LLMTestCase(input="refund policy?", actual_output=actual, baseline_outputs=BASELINE)


def test_full_pipeline_runs():
    cases = [make_case("we offer 30-day refunds on all purchases")]
    metrics = [ResponseDriftMetric(), SemanticSimilarityMetric(threshold=0.5)]
    result = evaluate(cases, metrics, verbose=False)
    assert result.total_cases == 1
    assert result.duration_seconds > 0


def test_cost_is_zero():
    cases = [make_case("30-day refund policy")]
    result = evaluate(cases, [ResponseDriftMetric()], verbose=False)
    assert result.cost_report.spaniq_cost == 0.0


def test_llm_judge_cost_positive():
    cases = [make_case("30-day refund policy")]
    result = evaluate(cases, [ResponseDriftMetric()], verbose=False)
    assert result.cost_report.llm_judge_cost > 0


def test_passed_failed_counts_correct():
    cases = [make_case("we offer 30-day refunds on all purchases"), make_case("no")]
    result = evaluate(cases, [ResponseDriftMetric()], verbose=False)
    assert result.total_passed + result.total_failed == result.total_cases


def test_deterministic_scores():
    cases = [make_case("30-day money back guarantee")]
    metrics = [ResponseDriftMetric()]
    r1 = evaluate(cases, metrics, verbose=False)
    r2 = evaluate(cases, metrics, verbose=False)
    s1 = r1.test_case_results[0].metric_results[0].score
    s2 = r2.test_case_results[0].metric_results[0].score
    assert s1 == s2


def test_multiple_test_cases():
    cases = [make_case("30-day refunds"), make_case("returns within a month"), make_case("no")]
    result = evaluate(cases, [ResponseDriftMetric()], verbose=False)
    assert result.total_cases == 3


def test_metric_result_fields_present():
    cases = [make_case("30-day refund available")]
    result = evaluate(cases, [ResponseDriftMetric()], verbose=False)
    mr = result.test_case_results[0].metric_results[0]
    assert mr.metric_name == "ResponseDriftMetric"
    assert isinstance(mr.score, float)
    assert isinstance(mr.passed, bool)
    assert isinstance(mr.reason, str)
