"""Tests for benchmark runner (Step 17) and analysis module (Step 19)."""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from benchmarks.config import DATASET_FILES
from benchmarks.runners.spaniq_runner import run_spaniq_eval, BenchmarkResult


QA_PATH = DATASET_FILES["qa_factual"]


def test_spaniq_runner_returns_benchmark_result():
    result = run_spaniq_eval(QA_PATH, n_runs=2)
    assert isinstance(result, BenchmarkResult)
    assert result.tool == "spaniq"
    assert len(result.runs) == 2


def test_spaniq_variance_is_zero():
    """Core claim: spanIQ scores are exactly deterministic across runs."""
    result = run_spaniq_eval(QA_PATH, n_runs=3)
    assert result.score_variance == pytest.approx(0.0, abs=1e-10)
    assert result.score_std == pytest.approx(0.0, abs=1e-10)


def test_spaniq_cost_is_zero():
    result = run_spaniq_eval(QA_PATH, n_runs=2)
    assert result.total_cost_usd == 0.0


def test_report_csv_saves_correctly():
    from benchmarks.analysis.report import save_csv
    result = run_spaniq_eval(QA_PATH, n_runs=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_csv([result], tmp)
        assert path.exists()
        content = path.read_text()
        assert "spaniq" in content
        assert "mean_score" in content


def test_ragas_runner_skips_without_deps(monkeypatch):
    """ragas runner raises (CLI-caught skip) when ragas/key are missing."""
    from benchmarks.runners.ragas_runner import run_ragas_eval

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    rag_path = DATASET_FILES["rag_retrieval"]
    with pytest.raises((ImportError, EnvironmentError)):
        run_ragas_eval(rag_path, n_runs=1)


def test_report_summary_md_contains_std_dev():
    from benchmarks.analysis.report import save_summary_md
    result = run_spaniq_eval(QA_PATH, n_runs=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_summary_md([result], tmp)
        content = path.read_text()
        assert "spaniq" in content.lower()
        assert "0.0000" in content
