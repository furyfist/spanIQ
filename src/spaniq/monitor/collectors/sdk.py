from __future__ import annotations

import time
from collections.abc import Iterator

from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace


class SDKCollector(BaseCollector):
    """In-process collector for direct SDK usage.

    Push traces in via ``ingest()``, then pass this collector to a Monitor.
    Call ``close()`` to signal no more traces will be pushed.
    """

    def __init__(self) -> None:
        self._queue: list[Trace] = []
        self._closed = False

    def name(self) -> str:
        return "sdk"

    def ingest(self, input: str, output: str, **kwargs) -> None:  # noqa: A002
        """Push a trace into the collector queue."""
        self._queue.append(Trace(input=input, output=output, **kwargs))

    def collect(self) -> Iterator[Trace]:
        while not self._closed or self._queue:
            if self._queue:
                yield self._queue.pop(0)
            else:
                time.sleep(0.05)

    def close(self) -> None:
        """Signal that no more traces will be ingested."""
        self._closed = True
