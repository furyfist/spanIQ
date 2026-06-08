from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from spaniq.monitor.trace import Trace


class BaseCollector(ABC):
    """Abstract base for trace collection from any source."""

    @abstractmethod
    def collect(self) -> Iterator[Trace]:
        """Yield traces from the source. May block waiting for new data."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable source name for logging."""
        ...
