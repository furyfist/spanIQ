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
| `gen_ai.content.prompt` | `Trace.input` |
| `gen_ai.content.completion` | `Trace.output` |
| Span duration | `ComponentSpan.latency_ms` |
| Span status ERROR | `ComponentSpan.error = True` |

Compatible with: `opentelemetry-instrumentation-openai`, `opentelemetry-instrumentation-anthropic`,
OpenLLMetry, Logfire, and any library that emits GenAI semconv v1.41+.

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

---

## Port configuration

```bash
# Custom ports
spaniq collect-otel --baseline my_baseline --grpc-port 4317 --http-port 4318

# Store only (no monitoring, just ingest)
spaniq collect-otel --store-only --db spaniq.db
```

Default ports follow the OTel standard: **4317** (gRPC), **4318** (HTTP/JSON).

---

## Troubleshooting

**Spans not arriving**
- Check your exporter endpoint: should be `http://localhost:4317` for gRPC (not `https`)
- Verify `spaniq collect-otel` is running before starting your app
- Try HTTP mode as a fallback: set exporter to `http://localhost:4318/v1/traces`

**Port already in use**
```bash
spaniq collect-otel --grpc-port 14317 --http-port 14318
```

**GenAI attributes missing**
Your OTel instrumentation library may use slightly different attribute names.
Use generic mode: add `spaniq.input` and `spaniq.output` attributes manually.

**Semconv version pinning**
spanIQ is pinned to GenAI semconv v1.41+ (development status). Attribute names
may change in future stable releases. Check `opentelemetry-semantic-conventions`
release notes when upgrading.
