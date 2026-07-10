"""Compute score variance across N identical benchmark runs."""

from __future__ import annotations

from benchmarks.runners.spaniq_runner import BenchmarkResult


def compute_variance(result: BenchmarkResult) -> float:
    return result.score_variance


def compute_std(result: BenchmarkResult) -> float:
    return result.score_std


def coefficient_of_variation(result: BenchmarkResult) -> float:
    mean = result.mean_score
    if mean == 0:
        return 0.0
    return result.score_std / mean
