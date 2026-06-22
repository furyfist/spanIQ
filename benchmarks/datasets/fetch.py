"""Download and cache benchmark datasets from HuggingFace.

Run once:  python -m benchmarks.datasets.fetch
Writes:
  benchmarks/datasets/qa_factual.jsonl         (50 rows from TriviaQA)
  benchmarks/datasets/summarization.jsonl      (30 rows from CNN/DailyMail)
  benchmarks/datasets/rag_retrieval.jsonl      (30 rows from RAGBench)
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"  wrote {len(rows)} rows → {path.name}")


def fetch_qa_factual(n: int = 50) -> None:
    """TriviaQA — factual QA pairs with verified short answers."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("error: 'datasets' not installed — run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
    rows = []
    for item in ds:
        if len(rows) >= n:
            break
        answers = item.get("answer", {})
        value = answers.get("value", "")
        if not value:
            continue
        rows.append({
            "input": item["question"],
            "reference_output": value,
            "output": f"The answer is {value}.",
            "category": "factual_qa",
            "source": "trivia_qa",
        })
    _write_jsonl(HERE / "qa_factual.jsonl", rows[:n])


def fetch_summarization(n: int = 30) -> None:
    """CNN/DailyMail — news article summarization with reference summaries."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(1)

    ds = load_dataset("cnn_dailymail", "3.0.0", split="validation", trust_remote_code=True)
    rows = []
    for item in ds:
        if len(rows) >= n:
            break
        article = item.get("article", "").strip()
        highlights = item.get("highlights", "").strip()
        if not article or not highlights:
            continue
        rows.append({
            "input": article[:800],
            "reference_output": highlights,
            "output": highlights,
            "category": "summarization",
            "source": "cnn_dailymail",
        })
    _write_jsonl(HERE / "summarization.jsonl", rows[:n])


def fetch_rag_retrieval(n: int = 30) -> None:
    """RAGBench — RAG scenarios with context, question, and expected answer."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(1)

    # rungalileo/ragbench has multiple subsets; use 'techqa' as it's compact
    try:
        ds = load_dataset("rungalileo/ragbench", "techqa", split="test", trust_remote_code=True)
    except Exception:
        # fallback: use hotpotqa if ragbench subset unavailable
        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        rows = []
        for item in ds:
            if len(rows) >= n:
                break
            rows.append({
                "input": item.get("question", ""),
                "reference_output": item.get("answer", ""),
                "output": item.get("answer", ""),
                "context": " ".join(
                    " ".join(s) for s in item.get("context", {}).get("sentences", [])
                )[:600],
                "category": "rag_retrieval",
                "source": "hotpot_qa",
            })
        _write_jsonl(HERE / "rag_retrieval.jsonl", rows[:n])
        return

    rows = []
    for item in ds:
        if len(rows) >= n:
            break
        rows.append({
            "input": item.get("question", ""),
            "reference_output": item.get("answer", ""),
            "output": item.get("answer", ""),
            "context": str(item.get("documents", [""])[:1])[:600],
            "category": "rag_retrieval",
            "source": "ragbench",
        })
    _write_jsonl(HERE / "rag_retrieval.jsonl", rows[:n])


def main() -> None:
    print("Fetching benchmark datasets from HuggingFace…")
    print("  qa_factual (TriviaQA)…")
    fetch_qa_factual(50)
    print("  summarization (CNN/DailyMail)…")
    fetch_summarization(30)
    print("  rag_retrieval (RAGBench/HotpotQA)…")
    fetch_rag_retrieval(30)
    print("Done. Datasets saved to benchmarks/datasets/")


if __name__ == "__main__":
    main()
