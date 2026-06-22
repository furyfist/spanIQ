"""spanIQ benchmark runner — deterministic, zero-cost evaluation."""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field


@dataclass
class RunResult:
    scores: list[float]
    time_sec: float
    cost_usd: float = 0.0


@dataclass
class BenchmarkResult:
    tool: str
    dataset: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        all_scores = [s for r in self.runs for s in r.scores]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    @property
    def score_variance(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        run_means = [sum(r.scores) / len(r.scores) for r in self.runs if r.scores]
        mean = sum(run_means) / len(run_means)
        return sum((x - mean) ** 2 for x in run_means) / len(run_means)

    @property
    def score_std(self) -> float:
        return self.score_variance ** 0.5

    @property
    def mean_time_sec(self) -> float:
        return sum(r.time_sec for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.runs)


def _load_dataset(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_spaniq_eval(dataset_path: str | pathlib.Path, n_runs: int = 5) -> BenchmarkResult:
    """Run spanIQ metrics on the dataset N times and measure variance and speed."""
    from spaniq.core.test_case import LLMTestCase
    from spaniq.core.evaluate import evaluate
    from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    metrics = [SemanticSimilarityMetric()]
    result = BenchmarkResult(tool="spaniq", dataset=path.stem)

    test_cases = [
        LLMTestCase(
            input=row["input"],
            actual_output=row["output"],
            expected_output=row.get("reference_output", row["output"]),
        )
        for row in rows
    ]

    for _ in range(n_runs):
        start = time.perf_counter()
        scores: list[float] = []
        eval_result = evaluate(test_cases, metrics, verbose=False)
        for tc_result in eval_result.test_case_results:
            for mr in tc_result.metric_results:
                scores.append(mr.score)
        elapsed = time.perf_counter() - start
        result.runs.append(RunResult(scores=scores, time_sec=elapsed, cost_usd=0.0))

    return result
