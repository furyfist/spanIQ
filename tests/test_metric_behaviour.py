"""Metric behaviour: calibrated consistency, graded stability, baseline cache, evaluate order."""

from __future__ import annotations

import threading

import pytest

from spaniq.core.evaluate import evaluate
from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.consistency import ConsistencyMetric, ks_critical_value
from spaniq.metrics.output_stability import OutputStabilityMetric
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
from spaniq.statistical import embeddings

REFUND_BASELINE = [
    "we offer 30-day refunds on all purchases",
    "refunds are available within 30 days of purchase",
    "you can return any item within 30 days for a full refund",
    "our refund policy allows returns within 30 days",
    "30-day money back guarantee on all items",
    "items can be refunded up to 30 days after you buy them",
]


def _case(actual: str, baseline=REFUND_BASELINE) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual, baseline_outputs=list(baseline))


def test_consistency_default_passes_on_paraphrase_and_fails_off_topic():
    """The old fixed 0.05 KS cut-off failed almost every output at these sample sizes."""
    ok = ConsistencyMetric()
    ok.measure(_case("refunds are accepted for 30 days after purchase"))
    assert ok.is_successful(), ok.reason

    bad = ConsistencyMetric()
    bad.measure(_case("quantum entanglement causes decoherence in neural networks"))
    assert not bad.is_successful(), bad.reason


def test_consistency_threshold_tracks_sample_size():
    metric = ConsistencyMetric()
    metric.measure(_case("refunds within 30 days"))
    n = len(REFUND_BASELINE)
    assert metric.threshold == pytest.approx(ks_critical_value(n, n * (n - 1) // 2, 0.05))
    assert ks_critical_value(30, 435, 0.05) < ks_critical_value(5, 10, 0.05)


def test_consistency_fixed_threshold_still_supported():
    metric = ConsistencyMetric(threshold=0.9)
    metric.measure(_case("refunds within 30 days"))
    assert metric.threshold == 0.9


def test_output_stability_is_zero_inside_baseline_range():
    score = OutputStabilityMetric().measure(_case("refunds are available within 30 days of buying"))
    assert score == 0.0


def test_output_stability_grades_rather_than_jumps():
    near = OutputStabilityMetric().measure(_case("refunds are available " * 3))
    far = OutputStabilityMetric().measure(_case("refunds are available " * 40))
    assert 0.0 < near < far <= 1.0


def test_output_stability_constant_baseline_feature_is_not_all_or_nothing():
    baseline = ["one two three four five."] * 5
    score = OutputStabilityMetric().measure(_case("one two three four five six.", baseline))
    assert 0.0 < score < 1.0


def test_baseline_embeddings_are_cached(monkeypatch):
    calls: list[int] = []
    real_embed = embeddings.embed

    def counting_embed(texts):
        calls.append(len(texts))
        return real_embed(texts)

    monkeypatch.setattr(embeddings, "embed", counting_embed)
    embeddings._reference_cache.clear()
    baseline = [f"baseline output number {i}" for i in range(20)]
    for _ in range(3):
        embeddings.embed_reference(baseline)
    assert calls == [20]


def test_semantic_similarity_unchanged_by_cache():
    tc = _case("refunds within thirty days")
    first = SemanticSimilarityMetric().measure(tc)
    second = SemanticSimilarityMetric().measure(tc)
    assert first == second


def test_model_loads_once_under_concurrency(monkeypatch):
    loads: list[str] = []

    class FakeModel:
        def __init__(self, name):
            loads.append(name)

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(embeddings, "_model", None)
    threads = [threading.Thread(target=embeddings.get_model) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1
    monkeypatch.setattr(embeddings, "_model", None)


def test_evaluate_keeps_duplicate_test_case_objects():
    tc = LLMTestCase(input="q", actual_output="Paris", expected_output="Paris")
    result = evaluate([tc, tc], [SemanticSimilarityMetric()], verbose=False)
    assert result.total_cases == 2
