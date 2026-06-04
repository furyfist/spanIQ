"""Realistic chatbot eval: 5 test cases across 3 metrics."""

from spaniq import LLMTestCase, evaluate
from spaniq.metrics import ConsistencyMetric, ResponseDriftMetric, SemanticSimilarityMetric

REFUND_BASELINES = [
    "we offer 30-day refunds on all purchases",
    "refunds are available within 30 days of purchase",
    "you can return any item within 30 days for a full refund",
    "our 30-day refund policy covers all standard purchases",
    "30-day money back guarantee on all items",
]

SHIPPING_BASELINES = [
    "standard shipping takes 3-5 business days",
    "orders ship within 3 to 5 business days",
    "expect delivery in 3-5 business days for standard shipping",
    "standard delivery is 3-5 working days",
    "shipping time is typically 3 to 5 business days",
]

test_cases = [
    LLMTestCase(
        input="what is your refund policy?",
        actual_output="we offer 30-day refunds on all purchases",
        baseline_outputs=REFUND_BASELINES,
    ),
    LLMTestCase(
        input="how long does shipping take?",
        actual_output="standard shipping takes 3-5 business days",
        baseline_outputs=SHIPPING_BASELINES,
    ),
    LLMTestCase(
        input="can I return a product?",
        actual_output="yes, returns are accepted within 30 days",
        expected_output="we offer 30-day refunds on all purchases",
        baseline_outputs=REFUND_BASELINES,
    ),
    LLMTestCase(
        input="what is the delivery timeframe?",
        actual_output="delivery typically takes 3-5 business days",
        baseline_outputs=SHIPPING_BASELINES,
    ),
    LLMTestCase(
        input="refund policy?",
        actual_output="30-day money back guarantee available for all orders",
        baseline_outputs=REFUND_BASELINES,
    ),
]

metrics = [
    ResponseDriftMetric(threshold=0.2),
    SemanticSimilarityMetric(threshold=0.6),
    ConsistencyMetric(threshold=0.6),
]

result = evaluate(test_cases, metrics)
