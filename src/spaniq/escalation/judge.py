"""LLM judges that escalation can call. A failed call raises; it never becomes a made-up score."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Protocol

DEFAULT_JUDGE_MODEL = os.environ.get("SPANIQ_JUDGE_MODEL", "openai/gpt-oss-120b")

PROMPT = """\
You review one answer from an LLM application.

Question:
{input}

Context the answer should be based on:
{context}

Answer:
{output}

On a scale from 1 to 10, how correct and fully supported by the context is the answer?
Reply with the number on the first line and one short sentence of reason on the second."""


class JudgeError(RuntimeError):
    """A judge call failed after retrying, or the reply held no score."""


class Judge(Protocol):
    name: str

    def judge(self, input: str, output: str, context: str | None) -> tuple[float, str]:  # noqa: A002
        """(score in [0, 1] where 1 is fully correct, one-line reason)."""
        ...


def parse_reply(raw: str) -> tuple[float, str]:
    match = re.search(r"\b(10|[1-9])\b", raw)
    if match is None:
        raise JudgeError(f"judge returned no score: {raw[:120]!r}")
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    reason = lines[1] if len(lines) > 1 else ""
    return (int(match.group(1)) - 1) / 9, reason


def _transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429))


def with_retry(
    call: Callable[[], str],
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return call()
        except Exception as exc:
            last = exc
            if not _transient(exc):
                break
            if i < attempts - 1:
                sleep(base_delay * 2**i)
    raise JudgeError(f"judge call failed: {last}") from last


class GroqJudge:
    """Judge backed by a Groq chat model. Needs the groq extra and GROQ_API_KEY."""

    name = "groq"

    def __init__(self, model: str | None = None, client=None, max_context_chars: int = 6000):
        self.model = model or DEFAULT_JUDGE_MODEL
        self.max_context_chars = max_context_chars
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise ImportError("groq SDK not installed — run: pip install spaniq[groq]") from exc
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise OSError("GROQ_API_KEY not set in environment")
            self._client = Groq(api_key=api_key)
        return self._client

    def judge(self, input: str, output: str, context: str | None) -> tuple[float, str]:  # noqa: A002
        client = self._get_client()
        prompt = PROMPT.format(
            input=input,
            context=(context or "(none recorded)")[: self.max_context_chars],
            output=output,
        )

        def call() -> str:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""

        return parse_reply(with_retry(call))


JUDGES: dict[str, Callable[[], Judge]] = {"groq": GroqJudge}


def make_judge(name: str) -> Judge:
    if name not in JUDGES:
        raise ValueError(f"unknown judge '{name}' (valid: {', '.join(JUDGES)})")
    return JUDGES[name]()
