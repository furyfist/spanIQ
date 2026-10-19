import numpy as np
import pytest

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.groundedness import GroundednessMetric, UnsupportedClaimsMetric
from spaniq.signals import extract
from spaniq.signals.nli import CrossEncoderNLI, check_groundedness, chunk_context, nli_signals

CONTEXT = "Paris is the capital of France. The Eiffel Tower was finished in 1889."


class WordOverlapNLI:
    """Entailed when every hypothesis word is in the premise; contradicted when it says 'not'."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        out = []
        for premise, hypothesis in pairs:
            words = {w.strip(".,").lower() for w in hypothesis.split()}
            known = {w.strip(".,").lower() for w in premise.split()}
            if "not" in words:
                out.append({"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05})
            elif words <= known:
                out.append({"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
            else:
                out.append({"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8})
        return out


def test_supported_sentences_score_high():
    g = check_groundedness("Paris is the capital of France.", CONTEXT, WordOverlapNLI())
    assert g.score == pytest.approx(0.95)
    assert g.unsupported == 0.0


def test_invented_sentence_is_unsupported():
    g = check_groundedness(
        "Paris is the capital of France. It hosts 40 million tourists yearly.",
        CONTEXT,
        WordOverlapNLI(),
    )
    assert g.unsupported == pytest.approx(0.5)
    assert g.sentences[1].entailment < 0.5


def test_contradiction_counted():
    g = check_groundedness("Paris is not the capital.", CONTEXT, WordOverlapNLI())
    assert g.contradicted == 1.0


def test_best_chunk_wins():
    context = "Unrelated text about whales. " * 60 + "Paris is the capital of France."
    assert len(chunk_context(context, max_chars=300)) > 1
    g = check_groundedness("Paris is the capital of France.", context, WordOverlapNLI(), 0.5, 300)
    assert g.score == pytest.approx(0.95)


def test_empty_output_is_trivially_grounded():
    assert check_groundedness("", CONTEXT, WordOverlapNLI()).score == 1.0


def test_one_batched_call():
    model = WordOverlapNLI()
    nli_signals("Paris is the capital. The tower is old. Lyon is big.", CONTEXT, model)
    assert model.calls == 1


def test_extract_uses_nli_only_when_given():
    assert "entailment" not in extract("Paris is the capital of France.", CONTEXT)
    s = extract("Paris is the capital of France.", CONTEXT, WordOverlapNLI())
    assert s["entailment"] == pytest.approx(0.95)


def test_groundedness_metric():
    metric = GroundednessMetric(model=WordOverlapNLI())
    tc = LLMTestCase(
        input="q", actual_output="Paris is the capital of France.", retrieval_context=[CONTEXT]
    )
    assert metric.measure(tc) == pytest.approx(0.95)
    assert metric.is_successful()
    bad = LLMTestCase(
        input="q", actual_output="Lyon hosts the Olympics.", retrieval_context=[CONTEXT]
    )
    metric.measure(bad)
    assert not metric.is_successful()
    assert "weakest" in metric.reason


def test_groundedness_requires_context():
    with pytest.raises(ValueError, match="retrieval_context"):
        GroundednessMetric(model=WordOverlapNLI()).measure(
            LLMTestCase(input="q", actual_output="a")
        )


def test_unsupported_claims_metric():
    metric = UnsupportedClaimsMetric()
    ok = LLMTestCase(
        input="q", actual_output="It was finished in 1889.", metadata={"context": CONTEXT}
    )
    bad = LLMTestCase(
        input="q", actual_output="It was finished in 1925.", metadata={"context": CONTEXT}
    )
    metric.measure(ok)
    assert metric.is_successful()
    metric.measure(bad)
    assert not metric.is_successful()


def test_cross_encoder_maps_labels_from_config(monkeypatch):
    import sentence_transformers

    class FakeCrossEncoder:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "C", (), {"id2label": {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}}
            )

        def predict(self, pairs, apply_softmax, batch_size):
            return np.array([[0.7, 0.2, 0.1]] * len(pairs))

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    nli = CrossEncoderNLI("fake-model")
    [probs] = nli.predict([("premise", "hypothesis")])
    assert probs == pytest.approx({"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1})
    assert nli.predict([]) == []


def test_metrics_selectable_from_cli():
    from spaniq.cli.runner import _metrics_from_names

    names = [m.name for m in _metrics_from_names("Groundedness,UnsupportedClaims")]
    assert names == ["GroundednessMetric", "UnsupportedClaimsMetric"]
