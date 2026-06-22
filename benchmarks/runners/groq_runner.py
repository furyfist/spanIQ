"""Groq-backed LLM-as-judge benchmark runner.

Uses Groq's LLaMA to judge output quality on a 1-10 scale.
Requires GROQ_API_KEY env var.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

from benchmarks.runners.spaniq_runner import BenchmarkResult, RunResult, _load_dataset


_JUDGE_PROMPT = """\
You are an impartial evaluator. Score the following model output on a scale from 1 to 10
based on how well it answers the question and matches the reference answer.

Question: {question}
Reference answer: {reference}
Model output: {output}

Respond with ONLY a single integer from 1 to 10. No explanation."""


def run_groq_eval(dataset_path: str | pathlib.Path, n_runs: int = 5,
                  model: str = "llama-3.3-70b-versatile") -> BenchmarkResult:
    """Run Groq LLM-as-judge evaluation N times on the dataset."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq SDK not installed — run: pip install spaniq[groq]")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment")

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
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                score = float(raw.split()[0]) / 10.0
                scores.append(max(0.0, min(1.0, score)))
                # Rough cost estimate: groq free tier but token counting for comparison
                usage = resp.usage
                cost += (usage.prompt_tokens * 0.27 + usage.completion_tokens * 0.27) / 1_000_000
            except Exception:
                scores.append(0.5)

            # Minimal rate-limit courtesy sleep
            time.sleep(0.05)

        elapsed = time.perf_counter() - start
        result.runs.append(RunResult(scores=scores, time_sec=elapsed, cost_usd=cost))
        print(f"    groq run {run_idx + 1}/{n_runs}: mean={sum(scores)/len(scores):.3f} t={elapsed:.1f}s")

    return result
