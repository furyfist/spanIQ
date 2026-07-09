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
class Prediction:
    """One scored item: its ground-truth label and the tool's raw 0-1 score.

    `score` is "how good the output looks" (high = good). The decision into a
    predicted label is applied later with a calibrated threshold, so the runner
    stores the raw score and never bakes in a cutoff.
    """
    item_id: int
    true_label: str          # "good" | "bad"
    score: float             # 0-1, higher = looks more correct
    failure_kind: str | None = None


@dataclass
class LabeledResult:
    """Accuracy-oriented result: predictions per run, plus determinism as a
    secondary stat. One entry in `runs` per identical repeat of the eval."""
    tool: str
    dataset: str
    runs: list[list[Prediction]] = field(default_factory=list)

    def scores_of_run(self, i: int) -> list[float]:
        return [p.score for p in self.runs[i]]

    @property
    def true_labels(self) -> list[str]:
        return [p.true_label for p in self.runs[0]] if self.runs else []

    @property
    def mean_scores(self) -> list[float]:
        """Per-item score averaged across runs (used for accuracy metrics)."""
        if not self.runs:
            return []
        n = len(self.runs[0])
        return [
            sum(run[i].score for run in self.runs) / len(self.runs)
            for i in range(n)
        ]

    @property
    def score_variance(self) -> float:
        """Determinism sidebar: variance of per-run mean scores."""
        if len(self.runs) < 2:
            return 0.0
        run_means = [sum(p.score for p in run) / len(run) for run in self.runs if run]
        mean = sum(run_means) / len(run_means)
        return sum((x - mean) ** 2 for x in run_means) / len(run_means)

    @property
    def score_std(self) -> float:
        return self.score_variance ** 0.5


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


def run_spaniq_predictions(dataset_path: str | pathlib.Path, n_runs: int = 5) -> "LabeledResult":
    """Score a labeled dataset with spanIQ and return per-item predictions.

    Uses the same deterministic embedding cosine metric, but reads each row's
    ground-truth `label` and emits a `Prediction` carrying the raw 0-1 score.
    The threshold that turns scores into decisions is chosen later (Phase 7),
    so nothing about accuracy is baked in here. Cost stays $0.00 — no LLM call.
    """
    from spaniq.core.test_case import LLMTestCase
    from spaniq.core.evaluate import evaluate
    from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

    path = pathlib.Path(dataset_path)
    rows = _load_dataset(path)
    metrics = [SemanticSimilarityMetric()]
    result = LabeledResult(tool="spaniq", dataset=path.stem)

    test_cases = [
        LLMTestCase(
            input=row["input"],
            actual_output=row["output"],
            expected_output=row.get("reference_output", row["output"]),
        )
        for row in rows
    ]

    for _ in range(n_runs):
        eval_result = evaluate(test_cases, metrics, verbose=False)
        preds: list[Prediction] = []
        for i, tc_result in enumerate(eval_result.test_case_results):
            score = tc_result.metric_results[0].score
            preds.append(Prediction(
                item_id=i,
                true_label=rows[i].get("label", "good"),
                score=score,
                failure_kind=rows[i].get("failure_kind"),
            ))
        result.runs.append(preds)

    return result
