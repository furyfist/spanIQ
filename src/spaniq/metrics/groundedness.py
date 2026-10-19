from __future__ import annotations

from spaniq.core.test_case import LLMTestCase
from spaniq.metrics.base import BaseMetric
from spaniq.signals.grounding import grounding_signals
from spaniq.signals.nli import NLIModel, check_groundedness, get_nli


def _context(test_case: LLMTestCase, metric: str) -> str:
    if test_case.retrieval_context:
        return "\n\n".join(test_case.retrieval_context)
    ctx = (test_case.metadata or {}).get("context")
    if ctx:
        return "\n\n".join(ctx) if isinstance(ctx, (list, tuple)) else str(ctx)
    raise ValueError(f"{metric} requires retrieval_context (or metadata['context'])")


class GroundednessMetric(BaseMetric):
    """Mean NLI entailment of the output's sentences by the retrieval context.

    Unlike the similarity metrics it needs no reference answer or baseline, only
    the context the answer was supposed to come from. score >= threshold passes.
    The NLI model loads on first measure(); pass model= to supply your own.
    """

    def __init__(self, threshold: float = 0.5, model: NLIModel | None = None):
        super().__init__(threshold=threshold)
        self.model = model

    def measure(self, test_case: LLMTestCase) -> float:
        context = _context(test_case, self.name)
        result = check_groundedness(
            test_case.actual_output, context, self.model or get_nli(), self.threshold
        )
        self.score = result.score
        weakest = min(result.sentences, key=lambda v: v.entailment, default=None)
        self.reason = (
            f"{result.unsupported:.0%} of sentences unsupported, "
            f"{result.contradicted:.0%} contradicted by the context"
            + (f"; weakest: {weakest.sentence[:80]!r}" if weakest else "")
        )
        return self.score

    def _check_threshold(self, score: float) -> bool:
        return score >= self.threshold


class UnsupportedClaimsMetric(BaseMetric):
    """Share of the output's numbers and names that its context never mentions.

    Lexical and model-free: catches invented figures and entities, not reworded
    false claims. score < threshold passes; the default fails on any unsupported one.
    """

    def __init__(self, threshold: float = 0.01):
        super().__init__(threshold=threshold)

    def measure(self, test_case: LLMTestCase) -> float:
        s = grounding_signals(test_case.actual_output, _context(test_case, self.name))
        self.score = max(s["unsupported_numbers"], s["unsupported_names"])
        self.reason = (
            f"unsupported numbers {s['unsupported_numbers']:.0%}, "
            f"unsupported names {s['unsupported_names']:.0%}"
        )
        return self.score

    def _check_threshold(self, score: float) -> bool:
        return score < self.threshold
