"""Langfuse-style LLM-as-judge benchmark runner.

Langfuse has no standalone local eval function — its LLM-as-a-Judge runs
server-side on ingested traces. This runner replicates that *methodology*
locally: the same kind of judge prompt template a Langfuse user configures,
run against Groq's LLaMA. It does not call any Langfuse server, so no Langfuse
SDK dependency is needed.

Requires: GROQ_API_KEY
"""
from __future__ import annotations

import os
import pathlib
import time

from benchmarks.runners.spaniq_runner import BenchmarkResult, RunResult, _load_dataset


# Mirrors a standard Langfuse LLM-as-a-Judge correctness template: structured
# input, 1-10 scale, JSON output with score + reasoning.
_JUDGE_PROMPT = """\
You are evaluating the quality of an AI assistant's response.

## Input
Question: {input}
Expected Answer: {expected}
Actual Answer: {output}

## Task
Rate the actual answer on a scale of 1-10 based on:
- Factual correctness compared to the expected answer
- Completeness of the response
- Relevance to the question

Respond with ONLY a JSON object:
{{"score": <number 1-10>, "reasoning": "<brief explanation>"}}"""


def _check_deps() -> None:
    """Raise if the Groq key needed for the judge call is missing."""
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set")
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError("openai SDK not installed — run: pip install openai") from exc


def _parse_score(text: str) -> float:
    """Parse a 0-1 score from the judge's JSON (or loose-number) response."""
    import json
    import re

    try:
        return min(max(float(json.loads(text)["score"]) / 10.0, 0.0), 1.0)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match:
            return min(float(match.group()) / 10.0, 1.0)
        return 0.5


async def _judge_one(client, model, row) -> tuple[float, int, int]:
    """Run one judge call; return (score 0-1, prompt_tokens, completion_tokens)."""
    prompt = _JUDGE_PROMPT.format(
        input=row["input"],
        expected=row.get("reference_output", ""),
        output=row["output"],
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return _parse_score(text), usage.prompt_tokens, usage.completion_tokens
    except Exception:
        return 0.5, 0, 0


async def _judge_batch(client, model, rows) -> list[tuple[float, int, int]]:
    """Judge all rows concurrently."""
    import asyncio

    return await asyncio.gather(*(_judge_one(client, model, r) for r in rows))
