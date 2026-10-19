# spanIQ

[![CI](https://github.com/furyfist/spaniq/actions/workflows/ci.yml/badge.svg)](https://github.com/furyfist/spaniq/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Deterministic LLM evaluation and production monitoring. $0/trace. No LLM judge.**

1,000 test cases × 3 metrics → **$0.00** and **no API calls**, instead of **9,000 LLM-judge calls**. Runs locally on CPU.

---

## Get your first signal in 2 minutes

```bash
# not on PyPI yet; install from GitHub
pip install "git+https://github.com/furyfist/spanIQ"
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
pip install "spaniq[otel] @ git+https://github.com/furyfist/spanIQ"
spaniq collect-otel --baseline my_baseline
# point your OTel exporter to localhost:4317 (gRPC) or localhost:4318/v1/traces (HTTP)
```

Then see your data:

```bash
pip install "spaniq[dashboard] @ git+https://github.com/furyfist/spanIQ"
spaniq dashboard
# opens http://localhost:8501
```

---

## The benchmark table

*Accuracy at catching bad outputs on a labeled 100-row QA set (positive class =
bad; thresholds calibrated on a held-out fold). Live run, 2026-10-19, judge
`openai/gpt-oss-120b` via Groq.*

| Tool | Precision | Recall | F1 | ROC-AUC | Cost / trace | Deterministic |
|------|-----------|--------|----|---------|--------------|---------------|
| Groq LLM judge | 1.000 | 1.000 | **1.000** | 1.000 | paid | usually, at temp 0 |
| deepeval (G-Eval) | 1.000 | 1.000 | **1.000** | 1.000 | paid | usually, at temp 0 |
| langfuse-style | 1.000 | 1.000 | **1.000** | 1.000 | paid | usually, at temp 0 |
| **spanIQ** | 0.806 | 0.967 | **0.879** | 0.857 | **$0.00** | ✅ by construction |

**The honest read:** at deciding whether one answer is right, an LLM judge beats
spanIQ's embedding similarity, and on the small summarization / RAG sets spanIQ
does worse than chance, because an answer that copies the reference and adds a
hallucination looks *more* similar than a correct paraphrase. spanIQ is not a
judge replacement. Its job is a deterministic, $0 signal of *change* against a
baseline (drift, structural shifts, which pipeline component broke first).

An earlier version of this table (2026-07-10) claimed spanIQ scored 1.0 on
summarization / RAG while the judges collapsed. That was wrong: the judge rows
were rate-limit fallbacks recorded as `0.5`, and spanIQ's score was a label leak.
Both are fixed. Full tables, the fairness rule, the correction and a per-item
audit trail: [`BENCHMARK_ACCURACY.md`](BENCHMARK_ACCURACY.md).

> The benchmark runs from a source checkout: `git clone https://github.com/furyfist/spanIQ && cd spanIQ && pip install -e ".[benchmark]"`, then
> `spaniq benchmark --tool spaniq --runs 5 --metric accuracy` (no key needed), or
> `spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 5 --metric accuracy` with `GROQ_API_KEY`.

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
- `OutputStabilityMetric` — how far an output's length and structure fall outside the baseline's range
- `ConsistencyMetric` — one-sided KS test: is an output further from the baseline than baseline outputs are from each other
- `ResponseDriftMetric` — vocabulary drift against the baseline (Population Stability Index)

### V2 — Production monitoring
- `Monitor` — stream traces from file, Langfuse, or SDK; score against baseline
- `TimelineStore` — SQLite-backed score time series with trend analysis
- `AlertEngine` — one alert per consecutive-failure incident, logged to JSONL and SQLite (shown in the dashboard)
- Collectors: `FileCollector`, `LangfuseCollector`, `SDKCollector`

### V3 — Root-cause attribution
- PELT changepoint detection per component
- CUSUM online alarms
- Cascade attribution: identifies root-cause component and downstream cascades
- `spaniq attribute` CLI command with terminal report and PNG export

### V4 — Ecosystem entry (this release)
- **OTelCollector** — OTLP/gRPC + HTTP receiver; auto-maps GenAI semconv spans
- **Streamlit dashboard** — 4 pages: Overview, Drift Timeline, Attribution, Alert Log
- **Accuracy benchmark** — compare spanIQ vs Groq/deepeval/ragas/langfuse on precision/recall/F1 at catching bad outputs (plus cost and determinism)
- **Quickstart overhaul** — 2-line first-signal experience
- **Generic span support** — `spaniq.*` attributes for non-GenAI apps

---

## Install

```bash
# Not on PyPI yet. From GitHub (add extras in brackets):
pip install "spaniq @ git+https://github.com/furyfist/spanIQ"           # core eval + monitoring
pip install "spaniq[viz] @ git+https://github.com/furyfist/spanIQ"      # PNG chart export (matplotlib)
pip install "spaniq[otel] @ git+https://github.com/furyfist/spanIQ"     # OTel receiver
pip install "spaniq[dashboard] @ git+https://github.com/furyfist/spanIQ" # local dashboard
pip install "spaniq[groq] @ git+https://github.com/furyfist/spanIQ"     # Groq baseline collection
pip install "spaniq[all] @ git+https://github.com/furyfist/spanIQ"      # everything

# Benchmark suite: needs a source checkout (datasets are not in the wheel)
git clone https://github.com/furyfist/spanIQ && cd spanIQ && pip install -e ".[benchmark]"
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
spaniq collect-otel --baseline my_baseline   # OTel receiver (localhost; --host 0.0.0.0 for remote)
spaniq collect-otel --store-only --traces-path otel_traces.jsonl   # ingest only, replay later

# Timeline inspection
spaniq timeline show --metric ResponseDriftMetric
spaniq timeline summary --metric SemanticSimilarityMetric
spaniq timeline export --metric ResponseDriftMetric --output chart.png

# Pipeline attribution (V3)
spaniq pipeline run --name my_pipeline --path traces.jsonl
spaniq pipeline status --name my_pipeline
spaniq attribute --pipeline my_pipeline --last 500

# Dashboard (V4)
spaniq dashboard                             # NEW
spaniq dashboard --db spaniq.db --port 8501

# Benchmark (V4)
spaniq benchmark --tool spaniq --runs 5 --metric accuracy      # NEW
spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 3 --metric accuracy
```

---

## Comparison

| Feature | spanIQ | deepeval | ragas | Phoenix | Langfuse |
|---|---|---|---|---|---|
| Deterministic by construction (no LLM call) | ✅ | ❌ | ❌ | ❌ | ❌ |
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

Apache 2.0, see [`LICENSE`](LICENSE).
