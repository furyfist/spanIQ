# Migrating from deepeval to spanIQ

spanIQ is not a full replacement for deepeval. It replaces the subset of deepeval metrics that use LLM-as-judge where a statistical equivalent exists. For metrics that genuinely require semantic judgment, keep using deepeval.

## Setup

```python
# deepeval
pip install deepeval

# spanIQ
pip install spaniq
```

## Test Case

```python
# deepeval
from deepeval.test_case import LLMTestCase
tc = LLMTestCase(input="...", actual_output="...", expected_output="...")

# spanIQ — same structure, adds baseline_outputs
from spaniq import LLMTestCase
tc = LLMTestCase(
    input="...",
    actual_output="...",
    expected_output="...",
    baseline_outputs=["known good output 1", "known good output 2", ...]
)
```

## Metric Mapping

| deepeval metric | spanIQ equivalent | Notes |
|---|---|---|
| `AnswerRelevancyMetric` | `SemanticSimilarityMetric` | Uses local embeddings, no API |
| `FaithfulnessMetric` | `SemanticSimilarityMetric` | Compare against retrieval_context |
| No equivalent | `ResponseDriftMetric` | Detects vocab/style drift from baseline |
| No equivalent | `OutputStabilityMetric` | Detects structural changes |
| No equivalent | `ConsistencyMetric` | Detects erratic output patterns |
| `HallucinationMetric` | No equivalent | Needs LLM judgment — keep deepeval |
| `ToxicityMetric` | No equivalent | Needs semantic safety model |
| `BiasMetric` | No equivalent | Needs LLM judgment |
| `SummarizationMetric` | No equivalent | Needs LLM judgment |
| `GEval` | No equivalent | Needs LLM judgment by design |

## Running Tests

```python
# deepeval
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

evaluate([tc], [AnswerRelevancyMetric(threshold=0.7)])

# spanIQ — identical signature
from spaniq import evaluate
from spaniq.metrics import SemanticSimilarityMetric

evaluate([tc], [SemanticSimilarityMetric(threshold=0.7)])
```

## pytest Integration

```python
# deepeval
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

def test_response():
    assert_test(tc, [AnswerRelevancyMetric(threshold=0.7)])

# spanIQ
from spaniq import assert_eval
from spaniq.metrics import SemanticSimilarityMetric

def test_response():
    assert_eval(tc, [SemanticSimilarityMetric(threshold=0.7)])
```

## Recommended Migration Strategy

1. Start by adding `baseline_outputs` to your existing test cases — collect 10–20 real outputs from your LLM when it was working correctly.
2. Replace `AnswerRelevancyMetric` with `SemanticSimilarityMetric` first — most direct swap.
3. Add `ResponseDriftMetric` to catch prompt or model changes early.
4. Keep deepeval for `HallucinationMetric`, `ToxicityMetric`, and `GEval` — these genuinely need LLM judgment.
5. Run both in CI until you have confidence in the statistical metrics.
