"""Groq-backed LLM-as-judge benchmark runner.

Uses Groq's LLaMA to judge output quality on a 1-10 scale.
Requires GROQ_API_KEY env var.
"""

from __future__ import annotations

import os
import pathlib
import time

from benchmarks.config import GROQ_MODEL
from benchmarks.runners._judge import score_from_ten, with_retry
from benchmarks.runners.spaniq_runner import (
    BenchmarkResult,
    LabeledResult,
    RunResult,
    _load_dataset,
    predictions_from_scores,
)

_JUDGE_PROMPT = """\
You are an impartial evaluator. Score the following model output on a scale from 1 to 10
based on how well it answers the question and matches the reference answer.

Question: {question}
Reference answer: {reference}
Model output: {output}

Respond with ONLY a single integer from 1 to 10. No explanation."""


def run_groq_eval(
    dataset_path: str | pathlib.Path, n_runs: int = 5, model: str = GROQ_MODEL
) -> BenchmarkResult:
    """Run Groq LLM-as-judge evaluation N times on the dataset."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("groq SDK not installed — run: pip install spaniq[groq]") from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise OSError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)
    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    result = BenchmarkResult(tool="groq-llm-judge", dataset=path.stem)

    for run_idx in range(n_runs):
        start = time.perf_counter()
        scores: list[float] = []
        cost = 0.0

        for row in rows:
            prompt = _JUDGE_PROMPT.format(
                question=row["input"],
                reference=row.get("reference_output", ""),
                output=row["output"],
            )
            resp = with_retry(
                lambda p=prompt: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": p}],
                    max_tokens=1024,
                    temperature=0.0,
                )
            )
            scores.append(score_from_ten(resp.choices[0].message.content or ""))
            # Groq is free tier; tokens are counted only for a cost comparison
            usage = resp.usage
            cost += (usage.prompt_tokens * 0.27 + usage.completion_tokens * 0.27) / 1_000_000

            # Minimal rate-limit courtesy sleep
            time.sleep(0.05)

        elapsed = time.perf_counter() - start
        result.runs.append(RunResult(scores=scores, time_sec=elapsed, cost_usd=cost))
        print(
            f"    groq run {run_idx + 1}/{n_runs}: mean={sum(scores) / len(scores):.3f} t={elapsed:.1f}s"
        )

    return result


def _score_row(client, model, row) -> float:
    """Judge one row, returning a 0-1 correctness score (raises JudgeError on failure)."""
    prompt = _JUDGE_PROMPT.format(
        question=row["input"],
        reference=row.get("reference_output", ""),
        output=row["output"],
    )
    resp = with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
    )
    return score_from_ten(resp.choices[0].message.content or "")


def run_groq_predictions(
    dataset_path: str | pathlib.Path, n_runs: int = 5, model: str = GROQ_MODEL
) -> LabeledResult:
    """Groq LLM-as-judge on a labeled dataset, returning per-item predictions."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("groq SDK not installed — run: pip install spaniq[groq]") from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise OSError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)
    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    result = LabeledResult(tool="groq-llm-judge", dataset=path.stem)

    for _ in range(n_runs):
        scores = []
        for row in rows:
            scores.append(_score_row(client, model, row))
            time.sleep(0.05)
        result.runs.append(predictions_from_scores(rows, scores))

    return result
