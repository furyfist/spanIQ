"""OTel collector: OTLP/gRPC + HTTP receiver that maps OTel spans to spanIQ Traces."""

from __future__ import annotations

import base64
import contextlib
import gzip
import json
import logging
import queue
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from spaniq.attribution.component import ComponentKind, ComponentSpan
from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTel GenAI semantic convention attribute keys
# ---------------------------------------------------------------------------
GENAI_SYSTEM = "gen_ai.system"
GENAI_PROVIDER = "gen_ai.provider.name"
GENAI_OPERATION = "gen_ai.operation.name"
GENAI_MODEL = "gen_ai.request.model"
# Current semconv carries content as JSON message lists in these attributes.
GENAI_INPUT_MESSAGES = "gen_ai.input.messages"
GENAI_OUTPUT_MESSAGES = "gen_ai.output.messages"
# Deprecated prompt/completion keys, still emitted by older instrumentations.
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
    def skip(cls) -> _ConvertedSpan:
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


def _iso_from_nanos(value) -> str:
    """OTLP start time (unix nanos, int or string) to an ISO-8601 UTC timestamp.

    Other collectors store ISO timestamps and the timeline sorts them as text,
    so raw nanosecond strings would sort wrongly against them.
    """
    try:
        nanos = int(value)
    except (TypeError, ValueError):
        return ""
    if nanos <= 0:
        return ""
    return datetime.fromtimestamp(nanos / 1e9, tz=timezone.utc).isoformat()


def _messages_text(raw, role: str | None = None) -> str:
    """Text of a gen_ai.*.messages JSON list; with role, only the last such message."""
    if not raw:
        return ""
    try:
        messages = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return str(raw)
    if not isinstance(messages, list):
        return ""
    if role is not None:
        matching = [m for m in messages if isinstance(m, dict) and m.get("role") == role]
        messages = matching[-1:]
    texts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("type", "text") == "text":
                texts.append(str(part.get("content", "")))
    return "\n".join(t for t in texts if t)


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
        return GENAI_OPERATION in attrs or GENAI_SYSTEM in attrs or GENAI_PROVIDER in attrs

    def _has_spaniq_attributes(self, attrs: dict) -> bool:
        return SPANIQ_INPUT in attrs and SPANIQ_OUTPUT in attrs

    # ------------------------------------------------------------------
    def _convert_genai(self, span: dict, attrs: dict) -> _ConvertedSpan:
        operation = attrs.get(GENAI_OPERATION, "chat")
        kind = OPERATION_TO_KIND.get(operation, ComponentKind.GENERATION)
        model = attrs.get(GENAI_MODEL, "unknown")
        system = attrs.get(GENAI_PROVIDER) or attrs.get(GENAI_SYSTEM, "genai")
        component_name = f"{system}.{model}"

        inp = _messages_text(attrs.get(GENAI_INPUT_MESSAGES), role="user")
        out = _messages_text(attrs.get(GENAI_OUTPUT_MESSAGES))
        ev_inp, ev_out = self._extract_events(span)
        inp = inp or ev_inp or attrs.get(GENAI_PROMPT, "")
        out = out or ev_out or attrs.get(GENAI_COMPLETION, "")

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
            timestamp=_iso_from_nanos(span.get("startTimeUnixNano")),
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
            timestamp=_iso_from_nanos(span.get("startTimeUnixNano")),
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
    """Assembles flat OTel spans into spanIQ Trace objects with ComponentSpans.

    Safe to call from the gRPC workers, the HTTP thread and the flush thread at
    once. With root_grace=0 a trace is emitted as soon as its root span
    arrives. A positive root_grace holds it that many seconds longer, because
    exporters batch independently and children often arrive after the root;
    flush_expired() then emits it. Spans that arrive for a trace that was
    already emitted are dropped instead of becoming a separate, wrongly rooted
    trace.
    """

    _EMITTED_MEMORY = 10_000

    def __init__(self, assembly_timeout: float = 5.0, root_grace: float = 0.0) -> None:
        self._pending: dict[str, _PendingTrace] = {}
        self._root_deadline: dict[str, float] = {}
        self._emitted: OrderedDict[str, None] = OrderedDict()
        self._timeout = assembly_timeout
        self._root_grace = root_grace
        self._lock = threading.Lock()

    def add_span(self, span: _ConvertedSpan) -> Trace | None:
        """Add a converted span; return the Trace if this completes one now."""
        if span.skipped or not span.trace_id:
            return None
        with self._lock:
            if span.trace_id in self._emitted:
                log.debug("dropping late span %s for emitted trace", span.span_id)
                return None
            bucket = self._pending.setdefault(span.trace_id, _PendingTrace())
            bucket.spans.append(span)
            if span.parent_span_id is not None:
                return None
            if self._root_grace <= 0:
                return self._assemble(span.trace_id)
            self._root_deadline[span.trace_id] = time.monotonic() + self._root_grace
            return None

    def flush_expired(self) -> list[Trace]:
        """Return Traces whose root grace has passed or that hit the assembly timeout."""
        now = time.monotonic()
        with self._lock:
            ready = [
                tid
                for tid, pt in self._pending.items()
                if now - pt.created_at > self._timeout
                or (tid in self._root_deadline and now >= self._root_deadline[tid])
            ]
            return [t for tid in ready if (t := self._assemble(tid)) is not None]

    # ------------------------------------------------------------------
    def _assemble(self, trace_id: str) -> Trace | None:
        """Caller holds the lock."""
        bucket = self._pending.pop(trace_id, None)
        self._root_deadline.pop(trace_id, None)
        self._emitted[trace_id] = None
        if len(self._emitted) > self._EMITTED_MEMORY:
            self._emitted.popitem(last=False)
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

        extra = {"timestamp": root.timestamp} if root.timestamp else {}
        return Trace(
            input=root.input,
            output=root.output,
            trace_id=trace_id,
            components=components or None,
            **extra,
        )


# ---------------------------------------------------------------------------
# OTLP HTTP handler — receives OTLP/HTTP JSON export requests
# ---------------------------------------------------------------------------
def _hex_ids(span: dict) -> dict:
    """MessageToDict renders proto bytes IDs as base64; OTLP/JSON uses hex."""
    for key in ("traceId", "spanId", "parentSpanId"):
        value = span.get(key)
        if value:
            with contextlib.suppress(ValueError, TypeError):
                span[key] = base64.b64decode(value).hex()
    return span


def _proto_request_to_dict(request) -> dict:
    """ExportTraceServiceRequest proto to the OTLP/JSON dict shape the converter reads."""
    from google.protobuf import json_format

    payload = json_format.MessageToDict(request)
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                _hex_ids(span)
    return payload


class _OTLPHTTPHandler:
    """OTLP/HTTP receiver for POST /v1/traces, JSON or protobuf, optionally gzipped."""

    PATH = "/v1/traces"

    def __init__(
        self, converter: SpanConverter, assembler: TraceAssembler, out: queue.Queue[Trace]
    ) -> None:
        self._converter = converter
        self._assembler = assembler
        self._out = out

    def handle(
        self, body: bytes, content_type: str = "application/json", content_encoding: str = ""
    ) -> tuple[int, str]:
        """Ingest one export request; return (HTTP status, error message or "")."""
        if content_encoding.lower() == "gzip":
            try:
                body = gzip.decompress(body)
            except OSError:
                return 400, "invalid gzip body"
        media = content_type.split(";")[0].strip().lower()
        if media == "application/x-protobuf":
            try:
                from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
            except ImportError:
                return 415, "protobuf needs opentelemetry-proto: pip install spaniq[otel]"
            request = trace_service_pb2.ExportTraceServiceRequest()
            try:
                request.ParseFromString(body)
            except Exception:
                return 400, "invalid protobuf payload"
            payload = _proto_request_to_dict(request)
        elif media in ("application/json", ""):
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return 400, "invalid JSON payload"
            if not isinstance(payload, dict):
                return 400, "payload must be an ExportTraceServiceRequest object"
        else:
            return 415, f"unsupported content type: {content_type}"
        self.ingest(payload)
        return 200, ""

    def ingest(self, payload: dict) -> None:
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

    Binds to localhost by default; pass host="0.0.0.0" to accept spans from
    other machines (there is no TLS or auth, so only on a trusted network).
    start() raises OSError if a port cannot be bound.

    Usage:
        collector = OTelCollector()
        collector.start()
        Monitor(baseline_name="my_baseline", collector=collector).run()
    """

    def __init__(
        self,
        grpc_port: int = 4317,
        http_port: int = 4318,
        assembly_timeout: float = 5.0,
        host: str = "127.0.0.1",
        root_grace: float = 1.0,
    ) -> None:
        self._grpc_port = grpc_port
        self._http_port = http_port
        self._host = host
        self._converter = SpanConverter()
        self._assembler = TraceAssembler(assembly_timeout, root_grace=root_grace)
        self._queue: queue.Queue[Trace] = queue.Queue()
        self._handler = _OTLPHTTPHandler(self._converter, self._assembler, self._queue)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._grpc_server = None
        self._httpd = None

    def name(self) -> str:
        return f"OTelCollector({self._host} grpc={self._grpc_port}, http={self._http_port})"

    def start(self) -> None:
        """Bind both receivers, then serve them from background threads."""
        self._httpd = self._build_http()
        try:
            self._grpc_server = self._build_grpc()
        except OSError:
            self._httpd.server_close()
            raise
        for name, target in (("otel-http", self._serve_http), ("otel-flush", self._run_flush)):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        if self._grpc_server is not None:
            self._grpc_server.start()
        log.info(
            "OTelCollector started on %s — gRPC :%d  HTTP :%d",
            self._host,
            self._grpc_port,
            self._http_port,
        )

    def stop(self) -> None:
        """Stop the receivers and background threads."""
        self._stop_event.set()
        if self._grpc_server is not None:
            self._grpc_server.stop(grace=2)

    def collect(self) -> Iterator[Trace]:
        """Yield Trace objects as they arrive. Blocks when queue is empty."""
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    # Background: periodically flush pending traces that are ready
    # ------------------------------------------------------------------
    def _run_flush(self) -> None:
        while not self._stop_event.wait(0.25):
            try:
                for trace in self._assembler.flush_expired():
                    self._queue.put(trace)
            except Exception:
                log.exception("OTel flush failed")

    # ------------------------------------------------------------------
    # OTLP/gRPC receiver
    # ------------------------------------------------------------------
    def _build_grpc(self):
        try:
            from concurrent import futures

            import grpc
            from opentelemetry.proto.collector.trace.v1 import (
                trace_service_pb2,
                trace_service_pb2_grpc,
            )
        except ImportError:
            log.warning(
                "gRPC deps not installed — OTLP/gRPC disabled. Run: pip install spaniq[otel]"
            )
            return None

        handler = self._handler

        class TraceServiceServicer(trace_service_pb2_grpc.TraceServiceServicer):
            def Export(self, request, context):  # noqa: N802
                handler.ingest(_proto_request_to_dict(request))
                return trace_service_pb2.ExportTraceServiceResponse()

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(TraceServiceServicer(), server)
        address = f"{self._host}:{self._grpc_port}"
        try:
            bound = server.add_insecure_port(address)
        except RuntimeError as exc:
            raise OSError(f"cannot bind OTLP/gRPC on {address}: {exc}") from exc
        if not bound:
            raise OSError(f"cannot bind OTLP/gRPC on {address}")
        return server

    # ------------------------------------------------------------------
    # OTLP/HTTP receiver
    # ------------------------------------------------------------------
    def _build_http(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        handler = self._handler

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if self.path.split("?")[0] != _OTLPHTTPHandler.PATH:
                    self._reply(404, "not found")
                    return
                length = int(self.headers.get("Content-Length", 0))
                status, error = handler.handle(
                    self.rfile.read(length),
                    content_type=self.headers.get("Content-Type", "application/json"),
                    content_encoding=self.headers.get("Content-Encoding", ""),
                )
                self._reply(status, error)

            def _reply(self, status: int, error: str) -> None:
                body = json.dumps({"error": error} if error else {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):  # silence default access log
                pass

        class Server(ThreadingHTTPServer):
            # On Windows SO_REUSEADDR lets a second server silently share a busy port
            allow_reuse_address = sys.platform != "win32"
            daemon_threads = True

        return Server((self._host, self._http_port), Handler)

    def _serve_http(self) -> None:
        httpd = self._httpd
        httpd.timeout = 0.5
        log.info("OTLP/HTTP listening on %s:%d", self._host, self._http_port)
        while not self._stop_event.is_set():
            httpd.handle_request()
        httpd.server_close()
