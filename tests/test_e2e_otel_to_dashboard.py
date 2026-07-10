"""E2E test: OTel span → OTelCollector → TraceAssembler → Trace → TimelineStore → dashboard query."""

from __future__ import annotations

import pytest

from spaniq.monitor.collectors.otel import (
    SpanConverter,
    TraceAssembler,
)
from spaniq.monitor.timeline_store import TimelineStore


def test_e2e_otel_span_to_timeline_store(tmp_path):
    """Full pipeline: OTel span → Trace → TimelineStore → dashboard-readable query."""
    db_path = str(tmp_path / "e2e.db")
    store = TimelineStore(db_path)

    # Simulate OTel span arriving
    converter = SpanConverter()
    assembler = TraceAssembler(assembly_timeout=1.0)

    span_dict = {
        "traceId": "e2etrace001",
        "spanId": "root001",
        "name": "chat",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "2000000000",
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4"}},
            {"key": "gen_ai.content.prompt", "value": {"stringValue": "What is 2+2?"}},
            {"key": "gen_ai.content.completion", "value": {"stringValue": "4"}},
        ],
        "status": {},
    }

    converted = converter.convert(span_dict)
    assert not converted.skipped
    trace = assembler.add_span(converted)
    assert trace is not None
    assert trace.input == "What is 2+2?"
    assert trace.output == "4"

    # Store a metric result into timeline (as Monitor would)
    store.record(
        trace_id=trace.trace_id,
        baseline_id="b1",
        metric_name="SemanticSimilarityMetric",
        score=0.92,
        threshold=0.7,
        passed=True,
        timestamp=trace.timestamp or "2024-01-01T00:00:00",
        component="default",
    )

    # Verify dashboard-level query works
    rows = store.query("SemanticSimilarityMetric", last_n=10)
    assert len(rows) == 1
    assert rows[0].score == pytest.approx(0.92)
    assert rows[0].passed is True
