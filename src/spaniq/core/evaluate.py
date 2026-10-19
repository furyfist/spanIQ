import copy
import time
from concurrent.futures import ThreadPoolExecutor

from spaniq.core.eval_result import EvalResult, MetricResult, TestCaseResult
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.reporting.cost_report import build_cost_report, print_report


def _run_single(metric: BaseMetric, test_case: LLMTestCase) -> MetricResult:
    score = metric.measure(test_case)
    return MetricResult(
        metric_name=metric.name,
        score=score,
        threshold=metric.threshold,
        passed=metric.is_successful(),
        reason=metric.reason,
    )


def evaluate(
    test_cases: list[LLMTestCase],
    metrics: list[BaseMetric],
    max_workers: int = 4,
    verbose: bool = True,
) -> EvalResult:
    """Run all metrics on all test cases in parallel. Deterministic. Zero API cost."""
    start = time.perf_counter()

    # A list, not a dict keyed by id(tc): the same test case object passed twice
    # must produce two results, in input order.
    tc_futures: list = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for tc in test_cases:
            futures = [pool.submit(_run_single, copy.deepcopy(m), tc) for m in metrics]
            tc_futures.append((tc, futures))

        test_case_results = []
        for tc, futures in tc_futures:
            metric_results = [f.result() for f in futures]
            test_case_results.append(TestCaseResult(test_case=tc, metric_results=metric_results))

    duration = time.perf_counter() - start
    total_passed = sum(1 for r in test_case_results if r.passed)
    total_failed = len(test_case_results) - total_passed

    eval_result = EvalResult(
        test_case_results=test_case_results,
        total_passed=total_passed,
        total_failed=total_failed,
        total_cases=len(test_case_results),
        cost_report=build_cost_report(len(test_cases), len(metrics)),
        duration_seconds=duration,
    )

    if verbose:
        print_report(eval_result)

    return eval_result
