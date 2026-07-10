"""Tests for OTel GenAI semantic convention → spanIQ mapping (Step 4)."""

from __future__ import annotations

import pytest

from spaniq.attribution.component import ComponentKind
from spaniq.monitor.collectors.otel import SpanConverter


def _make_span(
    operation: str,
    model: str = "gpt-4",
    system: str = "openai",
    prompt: str = "hello",
    completion: str = "world",
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    parent_id: str | None = None,
    status_code: str = "",
) -> dict:
    span = {
        "traceId": "abc123",
        "spanId": "span001",
        "name": "chat",
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": operation}},
            {"key": "gen_ai.system", "value": {"stringValue": system}},
            {"key": "gen_ai.request.model", "value": {"stringValue": model}},
            {"key": "gen_ai.content.prompt", "value": {"stringValue": prompt}},
            {"key": "gen_ai.content.completion", "value": {"stringValue": completion}},
        ],
        "status": {"code": status_code} if status_code else {},
    }
    if parent_id:
        span["parentSpanId"] = parent_id
    return span


class TestSpanConverterGenAI:
    def setup_method(self):
        self.conv = SpanConverter()

    def test_chat_operation_maps_to_generation(self):
        span = _make_span("chat")
        result = self.conv.convert(span)
        assert not result.skipped
        assert result.component_kind == ComponentKind.GENERATION

    def test_retrieval_operation_maps_to_retrieval(self):
        span = _make_span("retrieval")
        result = self.conv.convert(span)
        assert result.component_kind == ComponentKind.RETRIEVAL

    def test_execute_tool_maps_to_tool(self):
        span = _make_span("execute_tool")
        result = self.conv.convert(span)
        assert result.component_kind == ComponentKind.TOOL

    def test_invoke_agent_maps_to_agent(self):
        span = _make_span("invoke_agent")
        result = self.conv.convert(span)
        assert result.component_kind == ComponentKind.AGENT

    def test_input_output_extracted_from_attributes(self):
        span = _make_span("chat", prompt="What is 2+2?", completion="4")
        result = self.conv.convert(span)
        assert result.input == "What is 2+2?"
        assert result.output == "4"

    def test_latency_calculated_from_span_timing(self):
        span = _make_span("chat", start_ns=0, end_ns=500_000_000)
        result = self.conv.convert(span)
        assert result.latency_ms == pytest.approx(500.0)

    def test_error_status_mapped(self):
        span = _make_span("chat", status_code="STATUS_CODE_ERROR")
        result = self.conv.convert(span)
        assert result.error is True

    def test_missing_prompt_graceful_fallback(self):
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
            ],
            "status": {},
        }
        result = self.conv.convert(span)
        assert not result.skipped
        assert result.input == ""
        assert result.output == ""
