"""OTel collector: OTLP/gRPC + HTTP receiver that maps OTel spans to spanIQ Traces."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTel GenAI semantic convention attribute keys (semconv v1.41+)
# ---------------------------------------------------------------------------
GENAI_SYSTEM = "gen_ai.system"
GENAI_OPERATION = "gen_ai.operation.name"
GENAI_MODEL = "gen_ai.request.model"
GENAI_PROMPT = "gen_ai.content.prompt"
GENAI_COMPLETION = "gen_ai.content.completion"

# spanIQ custom attribute keys (generic/non-GenAI mode)
SPANIQ_INPUT = "spaniq.input"
SPANIQ_OUTPUT = "spaniq.output"
SPANIQ_COMPONENT = "spaniq.component"
SPANIQ_COMPONENT_KIND = "spaniq.component_kind"

# GenAI operation name → ComponentKind
OPERATION_TO_KIND: dict[str, ComponentKind] = {
    "chat": ComponentKind.GENERATION,
    "text_completion": ComponentKind.GENERATION,
    "retrieval": ComponentKind.RETRIEVAL,
    "execute_tool": ComponentKind.TOOL,
    "invoke_agent": ComponentKind.AGENT,
}

# spaniq.component_kind string → ComponentKind
STRING_TO_KIND: dict[str, ComponentKind] = {
    "chat": ComponentKind.GENERATION,
    "generation": ComponentKind.GENERATION,
    "retrieval": ComponentKind.RETRIEVAL,
    "execute_tool": ComponentKind.TOOL,
    "tool": ComponentKind.TOOL,
    "invoke_agent": ComponentKind.AGENT,
    "agent": ComponentKind.AGENT,
    "default": ComponentKind.DEFAULT,
}


# ---------------------------------------------------------------------------
# Internal representation of a converted span before assembly
# ---------------------------------------------------------------------------
@dataclass
class _ConvertedSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    input: str
    output: str
    component_name: str
    component_kind: ComponentKind
    latency_ms: float | None
    error: bool
    timestamp: str
    skipped: bool = False

    @classmethod
    def skip(cls) -> "_ConvertedSpan":
        return cls(
            trace_id="",
            span_id="",
            parent_span_id=None,
            input="",
            output="",
            component_name="",
            component_kind=ComponentKind.DEFAULT,
            latency_ms=None,
            error=False,
            timestamp="",
            skipped=True,
        )


# ---------------------------------------------------------------------------
# SpanConverter — converts a single OTel span dict into _ConvertedSpan
# ---------------------------------------------------------------------------
class SpanConverter:
    """Converts OTel span dicts (from OTLP JSON/proto) to _ConvertedSpan."""

    def convert(self, span: dict) -> _ConvertedSpan:
        attrs = self._attrs(span)
        if self._has_genai_attributes(attrs):
            return self._convert_genai(span, attrs)
        if self._has_spaniq_attributes(attrs):
            return self._convert_generic(span, attrs)
        return _ConvertedSpan.skip()

    # ------------------------------------------------------------------
    def _has_genai_attributes(self, attrs: dict) -> bool:
        return GENAI_OPERATION in attrs or GENAI_SYSTEM in attrs

    def _has_spaniq_attributes(self, attrs: dict) -> bool:
        return SPANIQ_INPUT in attrs and SPANIQ_OUTPUT in attrs

    # ------------------------------------------------------------------
    def _convert_genai(self, span: dict, attrs: dict) -> _ConvertedSpan:
        operation = attrs.get(GENAI_OPERATION, "chat")
        kind = OPERATION_TO_KIND.get(operation, ComponentKind.GENERATION)
        model = attrs.get(GENAI_MODEL, "unknown")
        component_name = f"{attrs.get(GENAI_SYSTEM, 'genai')}.{model}"

        # Input/output may live in span events (OTel GenAI convention)
        inp, out = self._extract_events(span)
        if not inp:
            inp = attrs.get(GENAI_PROMPT, "")
        if not out:
            out = attrs.get(GENAI_COMPLETION, "")

        return _ConvertedSpan(
            trace_id=span.get("traceId", ""),
            span_id=span.get("spanId", ""),
            parent_span_id=span.get("parentSpanId") or None,
            input=inp,
            output=out,
            component_name=component_name,
            component_kind=kind,
            latency_ms=self._latency_ms(span),
            error=self._is_error(span),
            timestamp=span.get("startTimeUnixNano", ""),
        )

    def _convert_generic(self, span: dict, attrs: dict) -> _ConvertedSpan:
        kind_str = attrs.get(SPANIQ_COMPONENT_KIND, "default").lower()
        kind = STRING_TO_KIND.get(kind_str, ComponentKind.DEFAULT)
        return _ConvertedSpan(
            trace_id=span.get("traceId", ""),
            span_id=span.get("spanId", ""),
            parent_span_id=span.get("parentSpanId") or None,
            input=attrs.get(SPANIQ_INPUT, ""),
            output=attrs.get(SPANIQ_OUTPUT, ""),
            component_name=attrs.get(SPANIQ_COMPONENT, span.get("name", "span")),
            component_kind=kind,
            latency_ms=self._latency_ms(span),
            error=self._is_error(span),
            timestamp=span.get("startTimeUnixNano", ""),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _attrs(span: dict) -> dict:
        raw = span.get("attributes", [])
        if isinstance(raw, dict):
            return raw
        # OTLP proto JSON encodes attributes as [{key, value: {stringValue: ...}}]
        result: dict = {}
        for item in raw:
            k = item.get("key", "")
            v = item.get("value", {})
            if "stringValue" in v:
                result[k] = v["stringValue"]
            elif "intValue" in v:
                result[k] = v["intValue"]
            elif "doubleValue" in v:
                result[k] = v["doubleValue"]
            elif "boolValue" in v:
                result[k] = v["boolValue"]
        return result

    @staticmethod
    def _extract_events(span: dict) -> tuple[str, str]:
        inp = out = ""
        for event in span.get("events", []):
            name = event.get("name", "")
            event_attrs = SpanConverter._attrs(event)
            if name == "gen_ai.content.prompt":
                inp = event_attrs.get("gen_ai.prompt", "")
            elif name == "gen_ai.content.completion":
                out = event_attrs.get("gen_ai.completion", "")
        return inp, out

    @staticmethod
    def _latency_ms(span: dict) -> float | None:
        start = span.get("startTimeUnixNano")
        end = span.get("endTimeUnixNano")
        if start and end:
            try:
                return (int(end) - int(start)) / 1_000_000
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def _is_error(span: dict) -> bool:
        status = span.get("status", {})
        return str(status.get("code", "")).upper() in ("STATUS_CODE_ERROR", "ERROR", "2")


# ---------------------------------------------------------------------------
# TraceAssembler — groups flat OTel spans into spanIQ Trace objects
# ---------------------------------------------------------------------------
@dataclass
class _PendingTrace:
    spans: list[_ConvertedSpan] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


class TraceAssembler:
    """Assembles flat OTel spans into spanIQ Trace objects with ComponentSpans."""

    def __init__(self, assembly_timeout: float = 5.0) -> None:
        self._pending: dict[str, _PendingTrace] = {}
        self._timeout = assembly_timeout

    def add_span(self, span: _ConvertedSpan) -> Trace | None:
        """Add a converted span. Returns a complete Trace if the trace root has been seen
        and all child spans are accounted for (heuristic: root received = flush now)."""
        if span.skipped or not span.trace_id:
            return None

        bucket = self._pending.setdefault(span.trace_id, _PendingTrace())
        bucket.spans.append(span)

        # If this span has no parent it's the root — assemble immediately
        if span.parent_span_id is None:
            return self._assemble(span.trace_id)
        return None

    def flush_expired(self) -> list[Trace]:
        """Return Traces whose spans have been waiting longer than timeout."""
        now = time.monotonic()
        expired = [tid for tid, pt in self._pending.items() if now - pt.created_at > self._timeout]
        return [t for tid in expired if (t := self._assemble(tid)) is not None]

    # ------------------------------------------------------------------
    def _assemble(self, trace_id: str) -> Trace | None:
        bucket = self._pending.pop(trace_id, None)
        if not bucket or not bucket.spans:
            return None

        root = next((s for s in bucket.spans if s.parent_span_id is None), bucket.spans[0])
        children = [s for s in bucket.spans if s.span_id != root.span_id]

        components = [
            ComponentSpan(
                name=s.component_name,
                kind=s.component_kind,
                output=s.output,
                latency_ms=s.latency_ms,
                error=s.error,
            )
            for s in children
        ]

        return Trace(
            input=root.input,
            output=root.output,
            timestamp=root.timestamp or None,  # type: ignore[arg-type]
            components=components or None,
        )


# ---------------------------------------------------------------------------
# OTLP HTTP handler — receives OTLP/HTTP JSON export requests
# ---------------------------------------------------------------------------
class _OTLPHTTPHandler:
    """Minimal OTLP/HTTP JSON receiver (POST /v1/traces)."""

    def __init__(self, converter: SpanConverter, assembler: TraceAssembler,
                 out: "queue.Queue[Trace]") -> None:
        self._converter = converter
        self._assembler = assembler
        self._out = out

    def handle(self, body: bytes) -> None:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            log.warning("OTel HTTP: invalid JSON payload")
            return
        for resource_spans in payload.get("resourceSpans", []):
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    converted = self._converter.convert(span)
                    trace = self._assembler.add_span(converted)
                    if trace:
                        self._out.put(trace)


# ---------------------------------------------------------------------------
# OTelCollector — public class, implements BaseCollector
# ---------------------------------------------------------------------------
class OTelCollector(BaseCollector):
    """Receives OTel spans via OTLP/gRPC and OTLP/HTTP, converts to spanIQ Traces.

    Usage:
        collector = OTelCollector()
        collector.start()
        for trace in collector.collect():
            monitor.observe(trace)
    """

    def __init__(
        self,
        grpc_port: int = 4317,
        http_port: int = 4318,
        assembly_timeout: float = 5.0,
    ) -> None:
        self._grpc_port = grpc_port
        self._http_port = http_port
        self._converter = SpanConverter()
        self._assembler = TraceAssembler(assembly_timeout)
        self._queue: queue.Queue[Trace] = queue.Queue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def name(self) -> str:
        return f"OTelCollector(grpc={self._grpc_port}, http={self._http_port})"

    def start(self) -> None:
        """Start OTLP/gRPC and OTLP/HTTP receivers in background threads."""
        t_grpc = threading.Thread(target=self._run_grpc, daemon=True, name="otel-grpc")
        t_http = threading.Thread(target=self._run_http, daemon=True, name="otel-http")
        t_flush = threading.Thread(target=self._run_flush, daemon=True, name="otel-flush")
        for t in (t_grpc, t_http, t_flush):
            t.start()
            self._threads.append(t)
        log.info("OTelCollector started — gRPC :%d  HTTP :%d", self._grpc_port, self._http_port)

    def stop(self) -> None:
        """Signal all background threads to stop."""
        self._stop_event.set()

    def collect(self) -> Iterator[Trace]:
        """Yield Trace objects as they arrive. Blocks when queue is empty."""
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    # Background: periodically flush timed-out pending traces
    # ------------------------------------------------------------------
    def _run_flush(self) -> None:
        while not self._stop_event.is_set():
            for trace in self._assembler.flush_expired():
                self._queue.put(trace)
            time.sleep(1.0)

    # ------------------------------------------------------------------
    # OTLP/gRPC receiver
    # ------------------------------------------------------------------
    def _run_grpc(self) -> None:
        try:
            from concurrent import futures

            import grpc
            from opentelemetry.proto.collector.trace.v1 import (
                trace_service_pb2,
                trace_service_pb2_grpc,
            )
            from opentelemetry.proto.trace.v1 import trace_pb2  # noqa: F401
            from google.protobuf import json_format
        except ImportError:
            log.warning("gRPC deps not installed — OTLP/gRPC disabled. Run: pip install spaniq[otel]")
            return

        converter = self._converter
        assembler = self._assembler
        out = self._queue

        class TraceServiceServicer(trace_service_pb2_grpc.TraceServiceServicer):
            def Export(self, request, context):  # noqa: N802
                # Convert proto to dict via JSON round-trip for simplicity
                as_dict = json_format.MessageToDict(request)
                for resource_spans in as_dict.get("resourceSpans", []):
                    for scope_spans in resource_spans.get("scopeSpans", []):
                        for span in scope_spans.get("spans", []):
                            converted = converter.convert(span)
                            trace = assembler.add_span(converted)
                            if trace:
                                out.put(trace)
                return trace_service_pb2.ExportTraceServiceResponse()

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(TraceServiceServicer(), server)
        server.add_insecure_port(f"[::]:{self._grpc_port}")
        server.start()
        log.info("OTLP/gRPC listening on :%d", self._grpc_port)
        self._stop_event.wait()
        server.stop(grace=2)

    # ------------------------------------------------------------------
    # OTLP/HTTP receiver
    # ------------------------------------------------------------------
    def _run_http(self) -> None:
        try:
            from http.server import BaseHTTPRequestHandler, HTTPServer
        except ImportError:
            return

        handler_obj = _OTLPHTTPHandler(self._converter, self._assembler, self._queue)
        stop = self._stop_event

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                handler_obj.handle(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, fmt, *args):  # silence default access log
                pass

        httpd = HTTPServer(("", self._http_port), Handler)
        httpd.timeout = 0.5
        log.info("OTLP/HTTP listening on :%d", self._http_port)
        while not stop.is_set():
            httpd.handle_request()
        httpd.server_close()
