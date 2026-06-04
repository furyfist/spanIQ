import pytest

from spaniq.core.eval_result import CostReport, EvalResult, MetricResult, TestCaseResult
from spaniq.core.test_case import LLMTestCase


def test_empty_input_raises():
    with pytest.raises(ValueError, match="input cannot be empty"):
        LLMTestCase(input="", actual_output="some output")


def test_empty_output_raises():
    with pytest.raises(ValueError, match="actual_output cannot be empty"):
        LLMTestCase(input="some input", actual_output="")


def test_defaults():
    tc = LLMTestCase(input="hello", actual_output="world")
    assert tc.expected_output is None
    assert tc.baseline_outputs == []
    assert tc.retrieval_context is None
    assert tc.metadata is None


def test_full_construction():
    tc = LLMTestCase(
        input="what is the refund policy?",
        actual_output="30-day refunds",
        expected_output="we offer 30-day refunds",
        baseline_outputs=["30-day refunds", "refunds within 30 days"],
        metadata={"model": "gpt-4o", "version": "1.0"},
    )
    assert tc.input == "what is the refund policy?"
    assert len(tc.baseline_outputs) == 2
    assert tc.metadata["model"] == "gpt-4o"


def test_metric_result_passed_flag():
    mr = MetricResult(
        metric_name="ResponseDriftMetric",
        score=0.05,
        threshold=0.1,
        passed=True,
        reason="PSI 0.05 < threshold 0.1",
    )
    assert mr.passed is True


def test_test_case_result_passed_all_metrics_pass():
    tc = LLMTestCase(input="q", actual_output="a")
    mr = MetricResult("M", 0.9, 0.7, True, "ok")
    tcr = TestCaseResult(test_case=tc, metric_results=[mr])
    assert tcr.passed is True


def test_test_case_result_failed_if_any_metric_fails():
    tc = LLMTestCase(input="q", actual_output="a")
    mr1 = MetricResult("M1", 0.9, 0.7, True, "ok")
    mr2 = MetricResult("M2", 0.3, 0.7, False, "below threshold")
    tcr = TestCaseResult(test_case=tc, metric_results=[mr1, mr2])
    assert tcr.passed is False


def test_cost_report_savings():
    report = CostReport(n_test_cases=100, n_metrics=3, spaniq_cost=0.0, llm_judge_cost=12.5)
    assert report.savings == 12.5


def test_eval_result_structure():
    tc = LLMTestCase(input="q", actual_output="a")
    mr = MetricResult("M", 0.8, 0.7, True, "ok")
    tcr = TestCaseResult(test_case=tc, metric_results=[mr])
    report = CostReport(n_test_cases=1, n_metrics=1, llm_judge_cost=0.5)
    result = EvalResult(
        test_case_results=[tcr],
        total_passed=1,
        total_failed=0,
        total_cases=1,
        cost_report=report,
        duration_seconds=0.1,
    )
    assert result.total_cases == 1
    assert result.cost_report.savings == 0.5
