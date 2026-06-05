from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterator

from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace

logger = logging.getLogger(__name__)


class LangfuseCollector(BaseCollector):
    """Polls the Langfuse Observations API v2 for new GENERATION spans.

    Requires ``langfuse>=4.0.0`` (``pip install spaniq[langfuse]``).
    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and optionally
    LANGFUSE_HOST from the environment via the SDK's default auth.

    Uses ``from_start_time`` as a cursor so only new observations are
    fetched on each poll cycle.
    """

    def __init__(self, poll_interval: int = 30, limit: int = 100) -> None:
        self.poll_interval = poll_interval
        self.limit = limit
        self._cursor: str | None = None  # ISO timestamp watermark

    def name(self) -> str:
        return "langfuse"

    def collect(self) -> Iterator[Trace]:
        try:
            from langfuse import get_client
        except ImportError as exc:
            raise ImportError(
                "langfuse SDK not installed — run: pip install spaniq[langfuse]"
            ) from exc

        client = get_client()

        while True:
            kwargs: dict = {
                "type": "GENERATION",
                "limit": self.limit,
            }
            if self._cursor:
                kwargs["from_start_time"] = self._cursor

            try:
                resp = client.api.observations.get_many(**kwargs)
            except Exception:
                logger.exception("LangfuseCollector: failed to fetch observations")
                time.sleep(self.poll_interval)
                continue

            observations = getattr(resp, "data", []) or []
            new_cursor: str | None = None

            for obs in observations:
                start_time = getattr(obs, "start_time", None)
                if start_time is not None:
                    ts = (
                        start_time.isoformat()
                        if hasattr(start_time, "isoformat")
                        else str(start_time)
                    )
                    if new_cursor is None or ts > new_cursor:
                        new_cursor = ts
                else:
                    ts = datetime.now(timezone.utc).isoformat()

                yield Trace(
                    input=str(getattr(obs, "input", "") or ""),
                    output=str(getattr(obs, "output", "") or ""),
                    trace_id=str(getattr(obs, "trace_id", "") or ""),
                    timestamp=ts,
                    metadata={
                        "model": getattr(obs, "model", None),
                        "langfuse_id": str(getattr(obs, "id", "") or ""),
                    },
                )

            if new_cursor:
                self._cursor = new_cursor

            if not observations:
                logger.debug("LangfuseCollector: no new observations, sleeping %ds", self.poll_interval)

            time.sleep(self.poll_interval)
