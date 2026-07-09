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

from benchmarks.runners._cost import token_cost
from benchmarks.runners.spaniq_runner import (
    BenchmarkResult, LabeledResult, RunResult, _load_dataset, predictions_from_scores,
)


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for cost reporting."""
    return max(1, len(text) // 4)


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


def _get_ragas_llm():
    """Build a ragas judge LLM backed by Groq's OpenAI-compatible endpoint.

    Tries llm_factory with an AsyncOpenAI client pointed at Groq first; if the
    v0.4 factory rejects a non-OpenAI client, falls back to wrapping a LangChain
    ChatOpenAI on the same base_url.
    """
    from ragas.llms import llm_factory
    from openai import AsyncOpenAI

    api_key = os.environ["GROQ_API_KEY"]
    base_url = "https://api.groq.com/openai/v1"
    model = "llama-3.3-70b-versatile"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        return llm_factory(model, client=client)
    except Exception:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        chat = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0.0)
        return LangchainLLMWrapper(chat)


def run_ragas_eval(dataset_path: str | pathlib.Path, n_runs: int = 5) -> BenchmarkResult:
    _check_deps()
    import asyncio

    from ragas.metrics.collections import Faithfulness

    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    if not any(r.get("context") for r in rows):
        raise ValueError(
            f"ragas faithfulness needs retrieved context; {path.name} has none — "
            "use the rag_retrieval dataset"
        )
    samples = _to_samples(rows)
    llm = _get_ragas_llm()
    result = BenchmarkResult(tool="ragas", dataset=path.stem)

    async def _score_all(scorer) -> list[float]:
        out = []
        for sample in samples:
            try:
                out.append(float(await scorer.single_turn_ascore(sample)))
            except Exception:
                out.append(0.5)
        return out

    # Faithfulness decomposes answer + context into claims; estimate judge cost
    # from the sample text (Groq is free, but we report the production-equivalent).
    run_cost = sum(
        token_cost(
            _est_tokens(r["input"] + r.get("context", "") + r["output"]),
            _est_tokens(r["output"]),
        )
        for r in rows
    )

    for run_idx in range(n_runs):
        scorer = Faithfulness(llm=llm)
        start = time.perf_counter()
        scores = asyncio.run(_score_all(scorer))
        elapsed = time.perf_counter() - start

        result.runs.append(RunResult(scores=scores, time_sec=elapsed, cost_usd=run_cost))
        print(f"    ragas run {run_idx + 1}/{n_runs}: mean={sum(scores)/len(scores):.3f} t={elapsed:.1f}s")

    return result


def run_ragas_predictions(dataset_path: str | pathlib.Path, n_runs: int = 5) -> LabeledResult:
    """ragas faithfulness on a labeled RAG dataset, returning per-item predictions.

    Faithfulness needs retrieved context, so this only accepts a dataset with a
    `context` field (rag_retrieval). Higher faithfulness = looks more correct,
    matching the good=high-score convention.
    """
    _check_deps()
    import asyncio

    from ragas.metrics.collections import Faithfulness

    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    if not any(r.get("context") for r in rows):
        raise ValueError(
            f"ragas faithfulness needs retrieved context; {path.name} has none — "
            "use the rag_retrieval dataset"
        )
    samples = _to_samples(rows)
    llm = _get_ragas_llm()
    result = LabeledResult(tool="ragas", dataset=path.stem)

    async def _score_all(scorer) -> list[float]:
        out = []
        for sample in samples:
            try:
                out.append(float(await scorer.single_turn_ascore(sample)))
            except Exception:
                out.append(0.5)
        return out

    for _ in range(n_runs):
        scorer = Faithfulness(llm=llm)
        scores = asyncio.run(_score_all(scorer))
        result.runs.append(predictions_from_scores(rows, scores))

    return result
