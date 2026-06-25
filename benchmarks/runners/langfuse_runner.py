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


def _check_deps() -> None:
    """Raise if the Groq key needed for the judge call is missing."""
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set")
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError("openai SDK not installed — run: pip install openai") from exc
