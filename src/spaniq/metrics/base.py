from abc import ABC, abstractmethod

from spaniq.core.test_case import LLMTestCase


class BaseMetric(ABC):
    """Abstract base for all spanIQ metrics. Interface-compatible with deepeval's BaseMetric."""

    def __init__(self, threshold: float, name: str | None = None):
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str = ""
        self.name = name or self.__class__.__name__

    @abstractmethod
    def measure(self, test_case: LLMTestCase) -> float:
        """Compute metric score. Must be deterministic. Sets self.score and self.reason."""
        ...

    def is_successful(self) -> bool:
        if self.score is None:
            raise ValueError("measure() must be called before is_successful()")
        return self._check_threshold(self.score)

    @abstractmethod
    def _check_threshold(self, score: float) -> bool: ...
