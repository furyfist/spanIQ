from dataclasses import dataclass, field

from spaniq.core.test_case import LLMTestCase


@dataclass
class CostReport:
    n_test_cases: int
    n_metrics: int
    spaniq_cost: float = 0.0
    llm_judge_cost: float = 0.0

    @property
    def savings(self) -> float:
        return self.llm_judge_cost - self.spaniq_cost


@dataclass
class MetricResult:
    metric_name: str
    score: float
    threshold: float
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class TestCaseResult:
    test_case: LLMTestCase
    metric_results: list[MetricResult]

    @property
    def passed(self) -> bool:
        return all(mr.passed for mr in self.metric_results)


@dataclass
class EvalResult:
    test_case_results: list[TestCaseResult]
    total_passed: int
    total_failed: int
    total_cases: int
    cost_report: CostReport
    duration_seconds: float
