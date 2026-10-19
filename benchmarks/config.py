"""Benchmark configuration — model settings and dataset paths."""

from __future__ import annotations

import os
import pathlib

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DRIFT_RESULTS_DIR = pathlib.Path(__file__).parent / "results_drift"
GROUNDING_RESULTS_DIR = pathlib.Path(__file__).parent / "results_grounding"

# Groq retired llama-3.3-70b-versatile (calls now 404 model_not_found), which is
# what the July 2026 judge rows were run with. Override with SPANIQ_JUDGE_MODEL.
GROQ_MODEL = os.environ.get("SPANIQ_JUDGE_MODEL", "openai/gpt-oss-120b")
N_RUNS_DEFAULT = 5

DATASET_FILES = {
    "qa_factual": DATASETS_DIR / "qa_factual.jsonl",
    "summarization": DATASETS_DIR / "summarization.jsonl",
    "rag_retrieval": DATASETS_DIR / "rag_retrieval.jsonl",
}
