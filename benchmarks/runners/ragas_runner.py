"""ragas benchmark runner — faithfulness via the v0.4 collections API + Groq.

Uses ragas v0.4's collections API (SingleTurnSample / EvaluationDataset /
Faithfulness from ragas.metrics.collections), judged by Groq's LLaMA through
the OpenAI-compatible endpoint.

Requires: pip install ragas>=0.4  +  GROQ_API_KEY
"""
from __future__ import annotations

import os
import pathlib
import time

from benchmarks.runners.spaniq_runner import BenchmarkResult, RunResult, _load_dataset


def _check_deps() -> None:
    """Raise ImportError/EnvironmentError if ragas v0.4 API or key is missing."""
    try:
        from ragas.metrics.collections import Faithfulness  # noqa: F401
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ragas v0.4 collections API not available — run: pip install 'ragas>=0.4'"
        ) from exc
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set")


def _to_samples(rows: list[dict]) -> list:
    """Convert dataset rows to ragas SingleTurnSample objects.

    rag_retrieval.jsonl fields: input, output, reference_output, context
    (context is a single string, wrapped into the required list form).
    """
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for row in rows:
        context = row.get("context", row.get("reference_output", ""))
        samples.append(SingleTurnSample(
            user_input=row["input"],
            response=row["output"],
            retrieved_contexts=[context],
            reference=row.get("reference_output", ""),
        ))
    return samples


def run_ragas_eval(dataset_path: str | pathlib.Path, n_runs: int = 5) -> BenchmarkResult:
    _check_deps()
    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    result = BenchmarkResult(tool="ragas", dataset=path.stem)
    # filled in across subsequent steps
    return result
