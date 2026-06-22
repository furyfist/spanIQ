"""E2E test: spanIQ benchmark runs with variance=0 and cost=0."""
from __future__ import annotations

import pytest

from benchmarks.config import DATASET_FILES
from benchmarks.runners.spaniq_runner import run_spaniq_eval


def test_e2e_benchmark_variance_zero():
    """Core social-media claim: spanIQ scores are exactly reproducible."""
    result = run_spaniq_eval(DATASET_FILES["qa_factual"], n_runs=3)
    assert result.score_variance == pytest.approx(0.0, abs=1e-10), (
        f"Expected zero variance, got {result.score_variance}"
    )


def test_e2e_benchmark_cost_zero():
    result = run_spaniq_eval(DATASET_FILES["qa_factual"], n_runs=2)
    assert result.total_cost_usd == 0.0
