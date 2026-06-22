# spanIQ

[![CI](https://github.com/furyfist/spaniq/actions/workflows/ci.yml/badge.svg)](https://github.com/furyfist/spaniq/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Deterministic LLM evaluation and production monitoring. $0/trace. No LLM judge.**

1,000 test cases × 3 metrics → **$0.00** and **4 seconds** instead of **$34.20** and **9,000 API calls**.

---

## Get your first signal in 2 minutes

```bash
pip install spaniq
```

```python
from spaniq.core.test_case import LLMTestCase
from spaniq.core.evaluate import evaluate
from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

tc = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France.",
    expected_output="Paris",
)
result = evaluate([tc], [SemanticSimilarityMetric()])
# Deterministic. Same score every run. $0.00.
```

Or with OTel (zero code changes to your app):

```bash
pip install "spaniq[otel]"
spaniq collect-otel --baseline my_baseline
# point your OTel exporter to localhost:4317
```

Then see your data:

```bash
pip install "spaniq[dashboard]"
spaniq dashboard
# opens http://localhost:8501
```

---

## The benchmark table

*Run 5 times on the same 20-question dataset. Lower variance = more trustworthy scores.*

| Tool | Mean Score | Std Dev | Cost per 100 evals | Deterministic? |
|------|-----------|---------|-------------------|---------------|
| **spanIQ** | 0.8731 | **0.0000** | **$0.00** | ✅ Yes |
| Groq LLM judge | 0.81 | ~0.04 | ~$0.001 | ❌ No |
| deepeval (G-Eval) | 0.76 | ~0.12 | ~$0.50 | ❌ No |
| ragas | 0.74 | ~0.09 | ~$0.40 | ❌ No |

> Run this yourself: `spaniq benchmark --tool spaniq --runs 5`
> With Groq key: `spaniq benchmark --tool spaniq,groq --runs 5`

---

## Architecture

```
                 ┌─────────────────────────────────────┐
                 │         Streamlit Dashboard          │  ← V4
                 │  reads SQLite, renders interactive   │
                 └──────────────┬──────────────────────┘
                                │ reads
     ┌──────────────────────────┼──────────────────────┐
     │                   SQLite DB                      │
     │  baselines │ timeline │ alerts │ components      │
     └──────────────────────────┬──────────────────────┘
                                │ writes
┌───────────────────────────────┼───────────────────────┐
│              V1-V3 Pipeline (untouched)                │
│  Metrics → Monitor → TimelineStore → Attribution      │
└───────────────────────────────┬───────────────────────┘
                                │ feeds
┌───────────┬───────────┬───────┴───┬───────────────────┐
│  File     │ Langfuse  │   SDK     │  OTelCollector    │  ← V4
│ Collector │ Collector │ Collector │  OTLP/gRPC+HTTP   │
└───────────┴───────────┴───────────┴───────────────────┘
```

V4 adds one collector at the bottom and one visualization layer at the top. The middle is untouched.

---

## Features

### V1 — Deterministic evaluation
- `SemanticSimilarityMetric` — cosine similarity with sentence-transformers
- `OutputStabilityMetric` — distributional consistency against a reference sample
- `ConsistencyMetric` — self-consistency across paraphrased prompts
- `ResponseDriftMetric` — statistical drift detection (PSI, KS, JS divergence)

### V2 — Production monitoring
- `Monitor` — stream traces from file, Langfuse, or SDK; score against baseline
- `TimelineStore` — SQLite-backed score time series with trend analysis
- `AlertEngine` — consecutive-failure alerts with JSONL + SQLite logging
- Collectors: `FileCollector`, `LangfuseCollector`, `SDKCollector`

### V3 — Root-cause attribution
- PELT changepoint detection per component
- CUSUM online alarms
- Cascade attribution: identifies root-cause component and downstream cascades
- `spaniq attribute` CLI command with terminal report and PNG export

### V4 — Ecosystem entry (this release)
- **OTelCollector** — OTLP/gRPC + HTTP receiver; auto-maps GenAI semconv spans
- **Streamlit dashboard** — 4 pages: Overview, Drift Timeline, Attribution, Alert Log
- **Determinism benchmark** — compare spanIQ vs Groq/deepeval/ragas on variance, cost, speed
- **Quickstart overhaul** — 2-line first-signal experience
- **Generic span support** — `spaniq.*` attributes for non-GenAI apps

---

## Install

```bash
# Core eval + monitoring
pip install spaniq

# OTel receiver
pip install "spaniq[otel]"

# Local dashboard
pip install "spaniq[dashboard]"

# Benchmark suite
pip install "spaniq[benchmark]"

# Groq baseline collection
pip install "spaniq[groq]"

# Everything
pip install "spaniq[all]"
```

---

## CLI reference

```bash
# Baseline management
spaniq baseline collect --name my_baseline --prompt "..." --source groq --n 30
spaniq baseline list
spaniq baseline show my_baseline

# Production monitoring
spaniq monitor run --baseline my_baseline --source file --path traces.jsonl
spaniq collect-otel --baseline my_baseline   # NEW: OTel receiver

# Timeline inspection
spaniq timeline show --metric ResponseDriftMetric
spaniq timeline summary --metric SemanticSimilarityMetric
spaniq timeline export --metric ResponseDriftMetric --output chart.png

# Pipeline attribution (V3)
spaniq pipeline run --name my_pipeline --path traces.jsonl
spaniq attribute --pipeline my_pipeline --last 500

# Dashboard (V4)
spaniq dashboard                             # NEW
spaniq dashboard --db spaniq.db --port 8501

# Benchmark (V4)
spaniq benchmark --tool spaniq --runs 5      # NEW
spaniq benchmark --tool spaniq,groq --runs 3
```

---

## Comparison

| Feature | spanIQ | deepeval | ragas | Phoenix | Langfuse |
|---|---|---|---|---|---|
| Deterministic scores | ✅ | ❌ | ❌ | ❌ | ❌ |
| $0 per eval | ✅ | ❌ | ❌ | ❌ | ❌ |
| Production monitoring | ✅ | ❌ | ❌ | ✅ | ✅ |
| Changepoint attribution | ✅ | ❌ | ❌ | ❌ | ❌ |
| OTel integration | ✅ | ❌ | ❌ | ✅ | ✅ |
| Local dashboard | ✅ | ❌ | ❌ | ✅ | ✅ |
| No cloud required | ✅ | ✅ | ✅ | ❌ | ❌ |
| No API key to evaluate | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Documentation

- [Quickstart](docs/quickstart.md) — 2-minute first signal
- [OTel integration](docs/otel-integration.md) — connect any OTel app
- [Dashboard guide](docs/dashboard.md) — navigate the Streamlit UI
- [V4 build plan](docs/plans/v4.md) — architecture and design decisions

---

## License

Apache 2.0
