"""Tests for TraceAssembler (Step 5) and OTelCollector (Step 6)."""
from __future__ import annotations

import time

import pytest

from spaniq.monitor.collectors.otel import (
    SpanConverter,
    TraceAssembler,
    _ConvertedSpan,
    ComponentKind,
)
from spaniq.attribution.component import ComponentKind as CK


def _cspan(trace_id: str = "t1", span_id: str = "s1",
           parent_id: str | None = None,
           inp: str = "in", out: str = "out",
           kind: ComponentKind = CK.GENERATION) -> _ConvertedSpan:
    return _ConvertedSpan(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent_id,
        input=inp, output=out, component_name="llm", component_kind=kind,
        latency_ms=100.0, error=False, timestamp="0",
    )


class TestTraceAssembler:
    def setup_method(self):
        self.asm = TraceAssembler(assembly_timeout=0.2)

    def test_single_span_trace(self):
        span = _cspan(parent_id=None)
        trace = self.asm.add_span(span)
        assert trace is not None
        assert trace.input == "in"
        assert trace.output == "out"

    def test_multi_span_trace_assembled_on_root(self):
        child = _cspan(span_id="child1", parent_id="root1")
        self.asm.add_span(child)
        root = _cspan(span_id="root1", parent_id=None, inp="root-in", out="root-out")
        trace = self.asm.add_span(root)
        assert trace is not None
        assert trace.input == "root-in"
        assert len(trace.components) == 1

    def test_out_of_order_child_before_root(self):
        self.asm.add_span(_cspan(span_id="c1", parent_id="r1"))
        self.asm.add_span(_cspan(span_id="c2", parent_id="r1"))
        root = _cspan(span_id="r1", parent_id=None, inp="root", out="answer")
        trace = self.asm.add_span(root)
        assert trace is not None
        assert len(trace.components) == 2

    def test_timeout_flush_returns_incomplete_trace(self):
        self.asm.add_span(_cspan(span_id="orphan", parent_id="never-arrives"))
        time.sleep(0.3)
        traces = self.asm.flush_expired()
        assert len(traces) == 1

    def test_skipped_span_ignored(self):
        from spaniq.monitor.collectors.otel import _ConvertedSpan
        skipped = _ConvertedSpan.skip()
        result = self.asm.add_span(skipped)
        assert result is None

    def test_empty_trace_id_ignored(self):
        span = _cspan(trace_id="", parent_id=None)
        result = self.asm.add_span(span)
        assert result is None
