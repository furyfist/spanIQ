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
