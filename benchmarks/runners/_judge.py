"""Shared policy for LLM judge calls: retry transient failures, never fabricate a score.

Earlier runners returned a neutral 0.5 when a call errored. Under rate limiting
that turned whole datasets into 0.5s that were then reported as judgments, so a
call that still fails after retrying now raises JudgeError and the tool is
skipped with a visible message instead.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

ATTEMPTS = 6
BASE_DELAY_SEC = 2.0


class JudgeError(RuntimeError):
    """A judge call failed for good, or returned something that is not a score."""


def _permanent(exc: Exception) -> bool:
    """A 4xx other than timeout/conflict/rate limit will fail the same way again."""
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429)


def with_retry(
    call: Callable[[], T],
    attempts: int | None = None,
    base_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run call(), retrying with exponential backoff, raising JudgeError at the end."""
    attempts = attempts or ATTEMPTS
    base_delay = BASE_DELAY_SEC if base_delay is None else base_delay
    last: Exception | None = None
    for i in range(attempts):
        try:
            return call()
        except Exception as exc:
            last = exc
            if _permanent(exc):
                break
            if i < attempts - 1:
                sleep(base_delay * 2**i)
    raise JudgeError(f"judge call failed after {i + 1} attempt(s): {last}") from last


async def awith_retry(
    call: Callable[[], Awaitable[T]],
    attempts: int | None = None,
    base_delay: float | None = None,
) -> T:
    """Async twin of with_retry."""
    attempts = attempts or ATTEMPTS
    base_delay = BASE_DELAY_SEC if base_delay is None else base_delay
    last: Exception | None = None
    for i in range(attempts):
        try:
            return await call()
        except Exception as exc:
            last = exc
            if _permanent(exc):
                break
            if i < attempts - 1:
                await asyncio.sleep(base_delay * 2**i)
    raise JudgeError(f"judge call failed after {i + 1} attempt(s): {last}") from last


def score_from_ten(raw: str) -> float:
    """Parse the first 1-10 integer into a 0-1 score, raising JudgeError if absent."""
    match = re.search(r"\b(10|[1-9])\b", raw)
    if match is None:
        raise JudgeError(f"judge returned no score: {raw!r}")
    return int(match.group(1)) / 10.0
