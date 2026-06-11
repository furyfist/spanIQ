"""Tests for ComponentSpan and ComponentKind."""
import pytest
from spaniq.attribution.component import ComponentKind, ComponentSpan


def test_component_kind_values():
    assert ComponentKind.RETRIEVAL == "retrieval"
    assert ComponentKind.TOOL == "execute_tool"
    assert ComponentKind.GENERATION == "chat"
    assert ComponentKind.AGENT == "invoke_agent"
    assert ComponentKind.DEFAULT == "default"


def test_component_span_defaults():
    span = ComponentSpan(name="retrieval", kind=ComponentKind.RETRIEVAL, output="some text")
    assert span.latency_ms is None
    assert span.error is False
    assert span.metadata is None


def test_component_span_full():
    span = ComponentSpan(
        name="gen",
        kind=ComponentKind.GENERATION,
        output="answer",
        latency_ms=123.4,
        error=False,
        metadata={"model": "llama"},
    )
    assert span.latency_ms == 123.4
    assert span.metadata["model"] == "llama"


def test_component_kind_from_string():
    assert ComponentKind("retrieval") == ComponentKind.RETRIEVAL
    assert ComponentKind("chat") == ComponentKind.GENERATION


def test_component_kind_invalid():
    with pytest.raises(ValueError):
        ComponentKind("unknown_kind")
