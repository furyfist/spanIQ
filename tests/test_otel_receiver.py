"""OTel receiver hardening: thread safety, IDs, timestamps, OTLP/HTTP formats and statuses."""

from __future__ import annotations

import gzip
import json
import queue
import threading
import time
import urllib.error
import urllib.request

import pytest

from spaniq.attribution.component import ComponentKind as CK
from spaniq.monitor.collectors.otel import (
    OTelCollector,
    SpanConverter,
    TraceAssembler,
    _ConvertedSpan,
    _OTLPHTTPHandler,
)


def _cspan(trace_id="t1", span_id="s1", parent_id=None, ts="2024-01-01T00:00:00+00:00"):
    return _ConvertedSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_id,
        input="in",
        output="out",
        component_name="llm",
        component_kind=CK.GENERATION,
        latency_ms=1.0,
        error=False,
        timestamp=ts,
    )


def _payload(trace_id="ab" * 16, span_id="cd" * 8, start="1700000000000000000"):
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": "chat",
                                "startTimeUnixNano": start,
                                "endTimeUnixNano": start,
                                "attributes": [
                                    {"key": "spaniq.input", "value": {"stringValue": "q"}},
                                    {"key": "spaniq.output", "value": {"stringValue": "a"}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }


def test_trace_keeps_otel_trace_id_and_iso_timestamp():
    converted = SpanConverter().convert(_payload()["resourceSpans"][0]["scopeSpans"][0]["spans"][0])
    assert converted.timestamp == "2023-11-14T22:13:20+00:00"
    trace = TraceAssembler().add_span(converted)
    assert trace.trace_id == "ab" * 16
    assert trace.timestamp == "2023-11-14T22:13:20+00:00"


def test_missing_start_time_falls_back_to_now():
    trace = TraceAssembler().add_span(_cspan(ts=""))
    assert trace.timestamp and trace.timestamp[:2] == "20"


def test_root_grace_waits_for_late_children_then_drops_stragglers():
    asm = TraceAssembler(assembly_timeout=5.0, root_grace=0.05)
    assert asm.add_span(_cspan(span_id="root")) is None
    asm.add_span(_cspan(span_id="child", parent_id="root"))
    time.sleep(0.1)
    (trace,) = asm.flush_expired()
    assert len(trace.components) == 1
    # a child for an already emitted trace must not become its own trace
    assert asm.add_span(_cspan(span_id="late", parent_id="root")) is None
    time.sleep(0.1)
    assert asm.flush_expired() == []


def test_assembler_is_thread_safe():
    asm = TraceAssembler(assembly_timeout=0.0)
    errors: list[Exception] = []

    def add(worker):
        try:
            for i in range(500):
                asm.add_span(_cspan(trace_id=f"{worker}-{i}", parent_id="never"))
        except Exception as exc:
            errors.append(exc)

    def flush():
        try:
            for _ in range(200):
                asm.flush_expired()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(w,)) for w in range(4)]
    threads.append(threading.Thread(target=flush))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_genai_input_output_messages_are_read():
    span = {
        "traceId": "t",
        "spanId": "s",
        "attributes": [
            {"key": "gen_ai.provider.name", "value": {"stringValue": "openai"}},
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {
                "key": "gen_ai.input.messages",
                "value": {
                    "stringValue": json.dumps(
                        [
                            {"role": "system", "parts": [{"type": "text", "content": "be nice"}]},
                            {"role": "user", "parts": [{"type": "text", "content": "hi?"}]},
                        ]
                    )
                },
            },
            {
                "key": "gen_ai.output.messages",
                "value": {
                    "stringValue": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}]}]
                    )
                },
            },
        ],
    }
    converted = SpanConverter().convert(span)
    assert converted.input == "hi?"
    assert converted.output == "hello"
    assert converted.component_name.startswith("openai.")


def _handler():
    out: queue.Queue = queue.Queue()
    return _OTLPHTTPHandler(SpanConverter(), TraceAssembler(), out), out


def test_http_handler_statuses():
    handler, out = _handler()
    body = json.dumps(_payload()).encode()
    assert handler.handle(body) == (200, "")
    assert (
        handler.handle(
            gzip.compress(json.dumps(_payload("ef" * 16)).encode()), "application/json", "gzip"
        )[0]
        == 200
    )
    assert handler.handle(b"{not json")[0] == 400
    assert handler.handle(b"x", content_encoding="gzip")[0] == 400
    assert handler.handle(b"x", content_type="text/plain")[0] == 415
    assert out.qsize() == 2


def test_http_handler_accepts_protobuf():
    pb = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    from google.protobuf import json_format

    # proto JSON reads IDs as base64, so build the binary IDs explicitly
    request = json_format.ParseDict(
        _payload(trace_id="", span_id=""), pb.ExportTraceServiceRequest()
    )
    span = request.resource_spans[0].scope_spans[0].spans[0]
    span.trace_id = bytes.fromhex("ab" * 16)
    span.span_id = bytes.fromhex("cd" * 8)
    handler, out = _handler()
    assert handler.handle(request.SerializeToString(), "application/x-protobuf") == (200, "")
    trace = out.get_nowait()
    assert trace.trace_id == "ab" * 16
    assert trace.input == "q"


def test_live_http_receiver_routes_and_reports_errors():
    collector = OTelCollector(grpc_port=0, http_port=0, root_grace=0.0)
    collector.start()
    try:
        port = collector._httpd.server_address[1]
        base = f"http://127.0.0.1:{port}"

        def post(path, data, ctype="application/json"):
            req = urllib.request.Request(base + path, data=data, headers={"Content-Type": ctype})
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status
            except urllib.error.HTTPError as exc:
                return exc.code

        assert post("/v1/traces", json.dumps(_payload()).encode()) == 200
        assert post("/v1/metrics", b"{}") == 404
        assert post("/v1/traces", b"garbage") == 400
        trace = next(collector.collect())
        assert trace.input == "q"
    finally:
        collector.stop()


def test_bind_failure_raises_at_start():
    first = OTelCollector(grpc_port=0, http_port=0)
    first.start()
    try:
        taken = first._httpd.server_address[1]
        with pytest.raises(OSError):
            OTelCollector(grpc_port=0, http_port=taken).start()
    finally:
        first.stop()
