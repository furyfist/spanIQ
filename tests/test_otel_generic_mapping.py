"""Tests for generic spaniq.* attribute OTel span mapping (Step 7)."""

from __future__ import annotations

from spaniq.attribution.component import ComponentKind
from spaniq.monitor.collectors.otel import SpanConverter


def _generic_span(
    inp: str = "in",
    out: str = "out",
    component: str | None = None,
    kind: str | None = None,
    parent_id: str | None = None,
) -> dict:
    attrs = [
        {"key": "spaniq.input", "value": {"stringValue": inp}},
        {"key": "spaniq.output", "value": {"stringValue": out}},
    ]
    if component:
        attrs.append({"key": "spaniq.component", "value": {"stringValue": component}})
    if kind:
        attrs.append({"key": "spaniq.component_kind", "value": {"stringValue": kind}})
    span: dict = {
        "traceId": "t1",
        "spanId": "s1",
        "name": "my-span",
        "attributes": attrs,
        "status": {},
    }
    if parent_id:
        span["parentSpanId"] = parent_id
    return span


class TestSpanConverterGeneric:
    def setup_method(self):
        self.conv = SpanConverter()

    def test_generic_span_with_all_attributes(self):
        span = _generic_span("question", "answer", component="retriever", kind="retrieval")
        result = self.conv.convert(span)
        assert not result.skipped
        assert result.input == "question"
        assert result.output == "answer"
        assert result.component_name == "retriever"
        assert result.component_kind == ComponentKind.RETRIEVAL

    def test_missing_optional_component_name_uses_span_name(self):
        span = _generic_span("q", "a")
        result = self.conv.convert(span)
        assert result.component_name == "my-span"

    def test_missing_optional_kind_defaults_to_default(self):
        span = _generic_span("q", "a")
        result = self.conv.convert(span)
        assert result.component_kind == ComponentKind.DEFAULT

    def test_span_without_either_convention_is_skipped(self):
        span = {"traceId": "t1", "spanId": "s1", "attributes": [], "status": {}}
        result = self.conv.convert(span)
        assert result.skipped

    def test_span_with_only_input_is_skipped(self):
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "attributes": [{"key": "spaniq.input", "value": {"stringValue": "hi"}}],
            "status": {},
        }
        result = self.conv.convert(span)
        assert result.skipped

    def test_mixed_trace_genai_spans_take_priority(self):
        span = {
            "traceId": "t1",
            "spanId": "s1",
            "name": "chat",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                {"key": "spaniq.input", "value": {"stringValue": "generic-input"}},
                {"key": "spaniq.output", "value": {"stringValue": "generic-output"}},
            ],
            "status": {},
        }
        result = self.conv.convert(span)
        assert result.component_kind == ComponentKind.GENERATION
