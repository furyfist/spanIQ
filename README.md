# spaniq

[![CI](https://github.com/furyfist/spaniq/actions/workflows/ci.yml/badge.svg)](https://github.com/furyfist/spaniq/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Deterministic LLM evaluation and production monitoring without LLM-as-judge. Zero API cost. Fully reproducible scores.

1,000 test cases × 3 metrics → **$0.00** and **4 seconds** instead of **$34.20** and **9,000 API calls**.

## Install

```bash
pip install spaniq

# optional: Langfuse trace ingestion
pip install spaniq[langfuse]

# optional: Groq-based baseline collection and demos
pip install spaniq[groq]
```

## V1 — Offline Evaluation

### Quickstart

```python
from spaniq import LLMTestCase, evaluate
from spaniq.metrics import ResponseDriftMetric, SemanticSimilarityMetric

tc = LLMTestCase(
    input="what is your refund policy?",
    actual_output="we offer a 30-day money back guarantee",
    baseline_outputs=[
        "we offer 30-day refunds on all purchases",
        "refunds are available within 30 days of purchase",
        "you can return any item within 30 days for a full refund",
    ],
)

result = evaluate([tc], [ResponseDriftMetric(), SemanticSimilarityMetric(threshold=0.6)])
```

Output:
```
 spaniq run complete
 1 test cases × 2 metrics
 spaniq cost:     $0.00 (0 API calls)
 LLM-judge equiv: ~$0.01 (6 calls)
 duration:        4.21s
 passed:          1/1
 failed:          0/1
```

### pytest Integration

```python
from spaniq import LLMTestCase, assert_eval
from spaniq.metrics import ResponseDriftMetric, SemanticSimilarityMetric

def test_refund_response():
    tc = LLMTestCase(
        input="refund policy?",
        actual_output="30-day refund available",
        baseline_outputs=["we offer 30-day refunds", "refunds within 30 days"],
    )
    assert_eval(tc, [ResponseDriftMetric(), SemanticSimilarityMetric(threshold=0.6)])
```

Run with:
```bash
spaniq test run tests/
```

## V2 — Production Monitoring

V2 adds continuous monitoring of live LLM outputs against stored baselines. Detects prompt injection, model swaps, and RAG breakage in real-time at $0/trace.

### 10-line example

```python
from spaniq.monitor import BaselineStore, Monitor
from spaniq.monitor.collectors.file import FileCollector

# 1. collect a baseline (once)
store = BaselineStore()
store.create(
    name="refund-v1",
    prompt="what is your refund policy?",
    outputs=["we offer 30-day refunds"] * 20,
)

# 2. run the monitor against your trace file
monitor = Monitor(
    baseline_name="refund-v1",
    collector=FileCollector("traces.jsonl"),
    alert_after=3,
)
report = monitor.run()
print(f"{report.total_traces} traces, {report.alerts_fired} alerts")
```

### Baseline collection CLI

```bash
# collect 50 baseline outputs from Groq
spaniq baseline collect --name refund-v1 --prompt "what is your refund policy?" --n 50

# list all baselines
spaniq baseline list

# inspect a baseline
spaniq baseline show refund-v1
```

### Monitor CLI

```bash
# run monitor against a JSONL trace file
spaniq monitor run --baseline refund-v1 --source file --path traces.jsonl

# run monitor polling Langfuse every 30s
spaniq monitor run --baseline refund-v1 --source langfuse --poll-interval 30

# custom alert threshold and metrics
spaniq monitor run --baseline refund-v1 --source file --path traces.jsonl \
  --alert-after 5 --metrics ResponseDrift,SemanticSimilarity
```

JSONL trace format:
```json
{"input": "what is your refund policy?", "output": "we offer 30-day refunds", "timestamp": "2024-01-01T00:00:00+00:00"}
```

### Timeline CLI

```bash
# terminal sparkline of recent scores
spaniq timeline show --metric ResponseDriftMetric --last 50

# export PNG chart
spaniq timeline export --metric ResponseDriftMetric --last 200 --output drift.png

# aggregate statistics
spaniq timeline summary --metric ResponseDriftMetric --last 200
```

### Replay Demos

Three reproducible demos that show the monitoring in action. Run offline with pre-generated fixtures (no API key needed):

```bash
# demo 1: prompt injection → pirate persona → vocabulary drift detected
spaniq demo prompt-injection --offline

# demo 2: model swap 70B → 8B → structural change detected
spaniq demo model-swap --offline

# demo 3: RAG retrieval failure → hedging words → semantic drift detected
spaniq demo rag-breakage --offline

# run all three
spaniq demo run-all --offline
```

With `GROQ_API_KEY` set, omit `--offline` to generate fresh outputs from the Groq free tier.

## Metrics

| Metric | Method | Detects | Requires |
|---|---|---|---|
| `ResponseDriftMetric` | PSI on word distributions | Vocabulary/style drift | `baseline_outputs` |
| `SemanticSimilarityMetric` | Cosine similarity via MiniLM | Semantic drift | `expected_output` or `baseline_outputs` |
| `OutputStabilityMetric` | JS divergence on structural features | Length/structure changes | `baseline_outputs` |
| `ConsistencyMetric` | KS test on embedding distances | Erratic output patterns | `baseline_outputs` (≥5) |

## Cost Comparison

| Tool | 1,000 cases × 3 metrics | Deterministic | Offline |
|---|---|---|---|
| deepeval / ragas | ~$34/run | No | No |
| **spaniq** | **$0.00** | **Yes** | **Yes** |

## Architecture

```
V1 (eval):
  LLMTestCase → metrics → evaluate() → EvalResult

V2 (monitoring, built on V1):
  Trace Source → Collector → LLMTestCase → Monitor → metrics → TimelineStore → AlertEngine
       ↑                          ↑
  langfuse API              BaselineStore
  JSONL file                (baseline_outputs)
  direct SDK
```

## When spanIQ Is Not the Right Tool

- Subjective quality judgment ("is this helpful?") — needs LLM
- Factual accuracy / hallucination detection — needs LLM
- Safety and toxicity — needs a safety model
- Zero-shot eval with no baselines — nothing to compare against

See [docs/WHY.md](docs/WHY.md) for the full argument.

## Migration from deepeval

See [docs/DEEPEVAL_MIGRATION.md](docs/DEEPEVAL_MIGRATION.md) for a side-by-side mapping.

## Contributing

```bash
git clone https://github.com/furyfist/spaniq
cd spaniq
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## License

Apache 2.0
