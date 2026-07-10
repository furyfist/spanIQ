"""Shared token-based cost estimate for LLM-judge runners.

Groq is free in practice, but we report what the same call would cost so the
benchmark table shows an honest 'in production this is non-zero' number.
Rate mirrors the groq_runner estimate ($0.27 / 1M tokens, both directions).
"""

from __future__ import annotations

_RATE_PER_MILLION = 0.27


def token_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single judge call from token usage."""
    return (prompt_tokens + completion_tokens) * _RATE_PER_MILLION / 1_000_000
