# spaniq

[![CI](https://github.com/furyfist/spaniq/actions/workflows/ci.yml/badge.svg)](https://github.com/furyfist/spaniq/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Deterministic LLM evaluation without LLM-as-judge. Zero API cost. Fully reproducible scores.

1,000 test cases × 3 metrics → **$0.00** and **4 seconds** instead of **$34.20** and **9,000 API calls**.

## Install

```bash
pip install spaniq
```

## Quickstart

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

## pytest Integration

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
