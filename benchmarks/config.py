"""Benchmark configuration — model settings and dataset paths."""
from __future__ import annotations

import pathlib

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

GROQ_MODEL = "llama-3.3-70b-versatile"
N_RUNS_DEFAULT = 5

DATASET_FILES = {
    "qa_factual": DATASETS_DIR / "qa_factual.jsonl",
    "summarization": DATASETS_DIR / "summarization.jsonl",
    "rag_retrieval": DATASETS_DIR / "rag_retrieval.jsonl",
}
