import json

import pytest

from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.monitor.trace import Trace
from spaniq.signals import (
    GROUNDING_SIGNALS,
    context_for,
    extract,
    grounding_signals,
    relevance_signals,
    trace_signals,
)

CONTEXT = (
    "The Greenfield Accord was signed by 195 nations in 2021. It pledges a 45% cut in "
    "carbon emissions by 2035 and funds renewable energy in developing countries."
)


def test_faithful_paraphrase_is_grounded():
    s = grounding_signals("195 nations signed the Greenfield Accord, pledging 45% cuts.", CONTEXT)
    assert s["unsupported_numbers"] == 0.0
    assert s["unsupported_names"] == 0.0
    assert s["novel_words"] < 0.3


def test_copied_answer_with_invented_claim_is_flagged():
    """The case the reference-similarity metrics got backwards."""
    faithful = grounding_signals("195 nations signed the Greenfield Accord.", CONTEXT)
    invented = grounding_signals(
        "195 nations signed the Greenfield Accord. Dr. Elena Marsh said 3,400 jobs followed.",
        CONTEXT,
    )
    for name in GROUNDING_SIGNALS:
        assert invented[name] > faithful[name]


def test_swapped_number_is_unsupported():
    s = grounding_signals("206 nations signed it.", CONTEXT)
    assert s["unsupported_numbers"] == 1.0


def test_number_formatting_is_normalised():
    assert (
        grounding_signals("It covers 8000 meters.", "Found below 8,000 meters.")[
            "unsupported_numbers"
        ]
        == 0.0
    )


def test_sentence_initial_capitals_are_not_names():
    assert grounding_signals("Carbon cuts were pledged.", CONTEXT)["unsupported_names"] == 0.0


def test_no_claims_means_zero():
    s = grounding_signals("yes", CONTEXT)
    assert s == {"unsupported_numbers": 0.0, "unsupported_names": 0.0, "novel_words": 0.0}


def test_relevance():
    assert relevance_signals("When was the Greenfield Accord signed?", CONTEXT)[
        "question_coverage"
    ] == pytest.approx(1.0)
    assert (
        relevance_signals("What causes the Northern Lights?", CONTEXT)["question_coverage"] == 0.0
    )


def test_extract_adds_grounding_only_with_context():
    assert "novel_words" not in extract("an answer")
    assert "novel_words" in extract("an answer", CONTEXT)


def test_context_for_prefers_metadata_then_sources():
    span = ComponentSpan("retrieval", ComponentKind.RETRIEVAL, "from span")
    assert context_for(Trace("q", "a", metadata={"context": ["one", "two"]})) == "one\n\ntwo"
    assert context_for(Trace("q", "a", components=[span])) == "from span"
    assert context_for(Trace("q", "a")) is None


def test_trace_signals_per_component():
    trace = Trace(
        input="When was the Greenfield Accord signed?",
        output="It was signed in 2021.",
        components=[
            ComponentSpan("retrieval", ComponentKind.RETRIEVAL, CONTEXT, latency_ms=120.0),
            ComponentSpan("generation", ComponentKind.GENERATION, "It was signed in 1999."),
        ],
    )
    units = trace_signals(trace)
    assert set(units) == {"output", "retrieval", "generation"}
    assert units["retrieval"]["question_coverage"] == pytest.approx(1.0)
    assert units["retrieval"]["latency_ms"] == 120.0
    assert units["generation"]["unsupported_numbers"] == 1.0
    assert units["output"]["unsupported_numbers"] == 0.0
    assert units["generation"]["error"] == 0.0


def test_trace_from_dict_round_trip():
    record = {
        "input": "q",
        "output": "a",
        "trace_id": "t1",
        "components": [
            {"name": "r", "kind": "retrieval", "output": "ctx", "error": True},
            {"name": "bad"},
        ],
    }
    trace = Trace.from_dict(json.loads(json.dumps(record)))
    assert trace.trace_id == "t1"
    assert len(trace.components) == 1
    assert trace.components[0].kind is ComponentKind.RETRIEVAL
    assert trace.components[0].error is True
    with pytest.raises(KeyError):
        Trace.from_dict({"input": "q"})
