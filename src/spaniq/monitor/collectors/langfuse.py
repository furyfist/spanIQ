from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from datetime import datetime, timezone

from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace

logger = logging.getLogger(__name__)

# Guards against a server that never reports its page count
_MAX_PAGES_PER_POLL = 1000


def _text(value, role: str | None = None) -> str:
    """Plain text from a Langfuse input/output, which may be a string, a message
    dict, or a chat message list. With role, a list yields the last message of
    that role. Anything else is JSON, never a Python repr."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        messages = [m for m in value if isinstance(m, dict)]
        if role is not None:
            matching = [m for m in messages if m.get("role") == role]
            messages = matching[-1:] or messages[-1:]
        if messages and all("content" in m for m in messages):
            return "\n".join(_text(m["content"]) for m in messages)
    if isinstance(value, dict):
        for key in ("content", "text", "output", "completion"):
            if key in value:
                return _text(value[key])
    return json.dumps(value, ensure_ascii=False, default=str)


def _iso(start_time) -> str | None:
    if start_time is None:
        return None
    return start_time.isoformat() if hasattr(start_time, "isoformat") else str(start_time)


class LangfuseCollector(BaseCollector):
    """Polls the Langfuse Observations API v2 for new GENERATION spans.

    Requires ``langfuse>=4.0.0`` (``pip install spaniq[langfuse]``).
    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and optionally
    LANGFUSE_HOST from the environment via the SDK's default auth.

    Uses ``from_start_time`` as a cursor so only new observations are fetched
    on each poll. Every page is read, so bursts larger than ``limit`` are not
    lost, and observations at the cursor timestamp that were already yielded
    are skipped (the cursor is inclusive).
    """

    def __init__(self, poll_interval: int = 30, limit: int = 100) -> None:
        self.poll_interval = poll_interval
        self.limit = limit
        self._cursor: str | None = None  # ISO timestamp watermark
        self._seen_at_cursor: set[str] = set()

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
            try:
                observations = self._fetch_all(client)
            except Exception:
                logger.exception("LangfuseCollector: failed to fetch observations")
                time.sleep(self.poll_interval)
                continue

            yield from self._new_traces(observations)

            if not observations:
                logger.debug(
                    "LangfuseCollector: no new observations, sleeping %ds",
                    self.poll_interval,
                )
            time.sleep(self.poll_interval)

    def _fetch_all(self, client) -> list:
        """Every page of observations since the cursor."""
        observations: list = []
        for page in range(1, _MAX_PAGES_PER_POLL + 1):
            kwargs: dict = {"type": "GENERATION", "limit": self.limit, "page": page}
            if self._cursor:
                kwargs["from_start_time"] = self._cursor
            resp = client.api.observations.get_many(**kwargs)
            data = list(getattr(resp, "data", []) or [])
            observations.extend(data)
            total_pages = getattr(getattr(resp, "meta", None), "total_pages", None)
            if len(data) < self.limit or (isinstance(total_pages, int) and page >= total_pages):
                break
        return observations

    def _new_traces(self, observations: list) -> Iterator[Trace]:
        """Yield unseen observations oldest first and advance the cursor."""
        keyed = [(_iso(getattr(obs, "start_time", None)), obs) for obs in observations]
        keyed.sort(key=lambda pair: pair[0] or "")
        for ts, obs in keyed:
            obs_id = str(getattr(obs, "id", "") or "")
            if ts is not None and ts == self._cursor and obs_id in self._seen_at_cursor:
                continue
            if ts is not None and (self._cursor is None or ts > self._cursor):
                self._cursor = ts
                self._seen_at_cursor = set()
            if ts is not None and ts == self._cursor:
                self._seen_at_cursor.add(obs_id)

            trace_kwargs = {}
            trace_id = getattr(obs, "trace_id", None) or obs_id
            if trace_id:
                trace_kwargs["trace_id"] = str(trace_id)
            yield Trace(
                input=_text(getattr(obs, "input", None), role="user"),
                output=_text(getattr(obs, "output", None)),
                timestamp=ts or datetime.now(timezone.utc).isoformat(),
                metadata={
                    "model": getattr(obs, "model", None),
                    "langfuse_id": obs_id,
                },
                **trace_kwargs,
            )
