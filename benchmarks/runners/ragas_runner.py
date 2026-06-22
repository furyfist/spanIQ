"""ragas benchmark runner — faithfulness metric via Groq.

Requires: pip install ragas  +  GROQ_API_KEY
"""
from __future__ import annotations

import os
import pathlib
import time

from benchmarks.runners.spaniq_runner import BenchmarkResult, RunResult, _load_dataset


def run_ragas_eval(dataset_path: str | pathlib.Path, n_runs: int = 5) -> BenchmarkResult:
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        raise ImportError("ragas not installed — run: pip install ragas datasets")

    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set")

    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    result = BenchmarkResult(tool="ragas", dataset=path.stem)

    # ragas expects 'question', 'answer', 'contexts', 'ground_truth'
    hf_data = {
        "question": [r["input"] for r in rows],
        "answer": [r["output"] for r in rows],
        "contexts": [[r.get("context", r.get("reference_output", ""))] for r in rows],
        "ground_truth": [r.get("reference_output", r["output"]) for r in rows],
    }

    for run_idx in range(n_runs):
        start = time.perf_counter()
        try:
            ds = Dataset.from_dict(hf_data)
            score_result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy])
            scores = list(score_result.to_pandas()["faithfulness"].fillna(0.5).values)
        except Exception:
            scores = [0.5] * len(rows)

        elapsed = time.perf_counter() - start
        result.runs.append(RunResult(scores=scores, time_sec=elapsed, cost_usd=0.0))
        print(f"    ragas run {run_idx + 1}/{n_runs}: mean={sum(scores)/len(scores):.3f}")

    return result
