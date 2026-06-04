"""RAG eval using SemanticSimilarity against retrieved context — no LLM judge."""

from spaniq import LLMTestCase, evaluate
from spaniq.metrics import OutputStabilityMetric, SemanticSimilarityMetric

ANSWER_BASELINES = [
    "neural networks learn by adjusting weights through backpropagation",
    "backpropagation updates weights in a neural network to minimize loss",
    "neural nets use gradient descent and backpropagation to learn from data",
    "learning in neural networks happens via weight updates through backprop",
    "weights are adjusted using backpropagation to reduce prediction error",
]

test_cases = [
    LLMTestCase(
        input="how do neural networks learn?",
        actual_output="neural networks adjust weights using backpropagation to minimize loss",
        expected_output="neural networks learn by adjusting weights through backpropagation",
        retrieval_context=[
            "Backpropagation is an algorithm used to train neural networks.",
            "Gradient descent updates weights to minimize the loss function.",
        ],
        baseline_outputs=ANSWER_BASELINES,
    ),
    LLMTestCase(
        input="what is backpropagation?",
        actual_output="backpropagation computes gradients and updates network weights",
        expected_output="backpropagation updates weights in a neural network to minimize loss",
        baseline_outputs=ANSWER_BASELINES,
    ),
    LLMTestCase(
        input="explain weight updates in neural nets",
        actual_output="weights are updated using gradient descent during training",
        baseline_outputs=ANSWER_BASELINES,
    ),
]

metrics = [
    SemanticSimilarityMetric(threshold=0.6),
    OutputStabilityMetric(threshold=0.9),
]

result = evaluate(test_cases, metrics)
