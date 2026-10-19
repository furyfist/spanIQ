from spaniq.core.eval_result import CostReport, EvalResult, MetricResult, TestCaseResult
from spaniq.core.evaluate import evaluate
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.metrics.consistency import ConsistencyMetric
from spaniq.metrics.groundedness import GroundednessMetric, UnsupportedClaimsMetric
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


# Monitoring, drift and attribution pull in SQLite stores, scipy and ruptures, so
# they load on first access instead of on "import spaniq".
_LAZY = {
    "Monitor": "spaniq.monitor.monitor",
    "Trace": "spaniq.monitor.trace",
    "BaselineStore": "spaniq.monitor.baseline_store",
    "TimelineStore": "spaniq.monitor.timeline_store",
    "AlertEngine": "spaniq.monitor.alerting",
    "FileCollector": "spaniq.monitor.collectors.file",
    "SDKCollector": "spaniq.monitor.collectors.sdk",
    "LangfuseCollector": "spaniq.monitor.collectors.langfuse",
    "OTelCollector": "spaniq.monitor.collectors.otel",
    "PipelineMonitor": "spaniq.attribution.pipeline_monitor",
    "attribute": "spaniq.attribution.attributor",
    "attribute_drift": "spaniq.attribution.drift_attribution",
    "trace_signals": "spaniq.signals.extract",
    "compare": "spaniq.drift.detector",
    "WindowDriftDetector": "spaniq.drift.detector",
    "diff_traces": "spaniq.drift.diff",
    "load_traces": "spaniq.drift.diff",
    "JudgeBudget": "spaniq.escalation.budget",
    "GroqJudge": "spaniq.escalation.judge",
    "escalate": "spaniq.escalation.escalate",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'spaniq' has no attribute {name!r}")


__all__ = [
    "evaluate",
    "assert_eval",
    "BaseMetric",
    "LLMTestCase",
    "EvalResult",
    "MetricResult",
    "TestCaseResult",
    "CostReport",
    "ResponseDriftMetric",
    "SemanticSimilarityMetric",
    "OutputStabilityMetric",
    "ConsistencyMetric",
    "GroundednessMetric",
    "UnsupportedClaimsMetric",
    *_LAZY,
]
