# OTel Integration Guide

spanIQ V4 includes an embedded OTLP receiver that accepts spans from any
OpenTelemetry-instrumented application and converts them to spanIQ `Trace` objects
for monitoring and drift detection.

---

## Quick start

```bash
pip install "spaniq[otel]"

# Create a baseline
spaniq baseline collect --name my_baseline --prompt "Your LLM prompt here" --n 30

# Start receiving spans
spaniq collect-otel --baseline my_baseline
# gRPC: localhost:4317  |  HTTP: localhost:4318
```

---

## Connecting your app

Configure your OTel SDK to export to spanIQ's receiver.

### Python (opentelemetry-sdk)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True))
)
trace.set_tracer_provider(provider)
```

### With opentelemetry-instrumentation-openai

```bash
pip install opentelemetry-instrumentation-openai
```

```python
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()
# All OpenAI calls now auto-emit GenAI semantic convention spans
# spanIQ picks these up automatically
```

---

## Span formats

### Mode 1 — GenAI semantic conventions (auto-mapped)

Any span with OTel GenAI semantic convention attributes is automatically converted:

| OTel attribute | spanIQ mapping |
|---|---|
| `gen_ai.operation.name: "chat"` | `ComponentKind.GENERATION` |
| `gen_ai.operation.name: "retrieval"` | `ComponentKind.RETRIEVAL` |
| `gen_ai.operation.name: "execute_tool"` | `ComponentKind.TOOL` |
| `gen_ai.operation.name: "invoke_agent"` | `ComponentKind.AGENT` |
| `gen_ai.input.messages` (last `user` message text) | `Trace.input` |
| `gen_ai.output.messages` (text parts) | `Trace.output` |
| `gen_ai.content.prompt` / `gen_ai.content.completion` (deprecated, still read) | fallback input / output |
| `gen_ai.provider.name` or `gen_ai.system` | component name prefix |
| OTel `traceId` | `Trace.trace_id` |
| `startTimeUnixNano` | `Trace.timestamp` (ISO-8601 UTC) |
| Span duration | `ComponentSpan.latency_ms` |
| Span status ERROR | `ComponentSpan.error = True` |

Compatible with: `opentelemetry-instrumentation-openai`, `opentelemetry-instrumentation-anthropic`,
OpenLLMetry, Logfire, and any library that emits the GenAI semantic conventions, current or
the older `gen_ai.content.*` form.

### Mode 2 — Generic `spaniq.*` attributes (manual)

For apps not using GenAI conventions, add these attributes to any OTel span:

```python
with tracer.start_as_current_span("my-llm-call") as span:
    span.set_attribute("spaniq.input", user_query)
    span.set_attribute("spaniq.output", llm_response)
    span.set_attribute("spaniq.component", "retriever")          # optional
    span.set_attribute("spaniq.component_kind", "retrieval")     # optional
```

Valid `spaniq.component_kind` values: `chat`, `generation`, `retrieval`,
`execute_tool`, `tool`, `invoke_agent`, `agent`, `default`.

---

## Multi-span traces (pipelines)

spanIQ reassembles multi-span OTel traces into `Trace` objects with `ComponentSpan` lists:

```
OTel trace
├── root span (input/output of the full pipeline) → Trace
│   ├── retriever span → ComponentSpan(kind=RETRIEVAL)
│   ├── llm span      → ComponentSpan(kind=GENERATION)
│   └── tool span     → ComponentSpan(kind=TOOL)
```

The root span (no parent) becomes the `Trace`. Child spans become `ComponentSpan` entries.
V3's changepoint attribution then runs per-component on the collected data.

Exporters batch spans independently, so a child can arrive after its root. The collector
holds a trace for one second after its root arrives (`root_grace`) to pick up late children,
and gives up on a trace with no root after `assembly_timeout` (5 s). A span that arrives after
its trace was emitted is dropped rather than turned into a second, wrongly rooted trace.

---

## Port configuration

```bash
# Custom ports
spaniq collect-otel --baseline my_baseline --grpc-port 4317 --http-port 4318

# Store only (no monitoring, just ingest)
spaniq collect-otel --store-only --db spaniq.db

# Accept spans from other machines (no TLS or auth: trusted networks only)
spaniq collect-otel --baseline my_baseline --host 0.0.0.0
```

Default ports follow the OTel standard: **4317** (gRPC), **4318** (HTTP). The receiver binds
to `127.0.0.1` unless you pass `--host`. OTLP/HTTP accepts both `application/x-protobuf` (the
exporter default) and `application/json`, gzipped or not, on `POST /v1/traces`; malformed
payloads get a 4xx so the exporter reports the failure instead of silently losing spans.

---

## Troubleshooting

**Spans not arriving**
- Check your exporter endpoint: should be `http://localhost:4317` for gRPC (not `https`)
- Verify `spaniq collect-otel` is running before starting your app
- Try HTTP mode as a fallback: set exporter to `http://localhost:4318/v1/traces`

**Port already in use**
`spaniq collect-otel` exits with an error naming the port. Pick free ones:
```bash
spaniq collect-otel --grpc-port 14317 --http-port 14318
```

**GenAI attributes missing**
Your OTel instrumentation library may use slightly different attribute names.
Use generic mode: add `spaniq.input` and `spaniq.output` attributes manually.

**Semconv versions**
The GenAI conventions are still in development status and attribute names have changed
before. spanIQ reads both the current `gen_ai.input.messages` / `gen_ai.output.messages`
attributes and the older `gen_ai.content.*` keys. Check `opentelemetry-semantic-conventions`
release notes when upgrading.
