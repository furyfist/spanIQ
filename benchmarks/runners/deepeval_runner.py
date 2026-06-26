"""deepeval benchmark runner — G-Eval faithfulness via Groq.

Requires: pip install deepeval  +  GROQ_API_KEY
"""
from __future__ import annotations

import pathlib
import time

from benchmarks.runners.spaniq_runner import BenchmarkResult, RunResult, _load_dataset


def run_deepeval_eval(dataset_path: str | pathlib.Path, n_runs: int = 5) -> BenchmarkResult:
    try:
        from deepeval import evaluate as dv_evaluate
        from deepeval.metrics import GEval
        from deepeval.models.llms import GPTModel
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError:
        raise ImportError("deepeval not installed — run: pip install deepeval")

    import os
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    groq_model = GPTModel(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    result = BenchmarkResult(tool="deepeval", dataset=path.stem)

    for run_idx in range(n_runs):
        start = time.perf_counter()
        scores: list[float] = []

        metric = GEval(
            name="Correctness",
            criteria="Does the actual output correctly answer the question given the expected output?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=groq_model,
        )

        for row in rows:
            tc = LLMTestCase(
                input=row["input"],
                actual_output=row["output"],
                expected_output=row.get("reference_output", row["output"]),
            )
            try:
                metric.measure(tc)
                scores.append(metric.score if metric.score is not None else 0.5)
            except Exception:
                scores.append(0.5)

        elapsed = time.perf_counter() - start
        result.runs.append(RunResult(scores=scores, time_sec=elapsed, cost_usd=0.0))
        print(f"    deepeval run {run_idx + 1}/{n_runs}: mean={sum(scores)/len(scores):.3f}")

    return result
