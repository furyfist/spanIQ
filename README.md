# spaniq

Deterministic LLM evaluation without LLM-as-judge. Zero API cost, fully reproducible scores.

## Install

```bash
pip install spaniq
```

## Quickstart

```python
from spaniq import evaluate, LLMTestCase
from spaniq.metrics import ResponseDriftMetric, SemanticSimilarityMetric

tc = LLMTestCase(
    input="what is your refund policy?",
    actual_output="we offer a 30-day money back guarantee",
    baseline_outputs=[
        "we offer 30-day refunds on all purchases",
        "refunds are available within 30 days of purchase",
    ],
)

result = evaluate([tc], [ResponseDriftMetric(), SemanticSimilarityMetric()])
```
