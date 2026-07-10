"""CLI entry point for the benchmark suite.

The default metric is accuracy: precision / recall / F1 / AUC at catching bad
outputs on labeled data. The legacy determinism metric (score variance across
identical runs) remains available via `--metric variance`.

Usage (via spaniq CLI):
    spaniq benchmark --tool spaniq --runs 5
    spaniq benchmark --tool spaniq,groq,deepeval,ragas,langfuse --runs 5
    spaniq benchmark --tool spaniq --runs 5 --metric variance
    spaniq benchmark --setup   # download full HF datasets

Direct:
    python -m benchmarks.run_benchmark --tool spaniq --runs 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from benchmarks.config import DATASET_FILES, RESULTS_DIR
from benchmarks.runners.spaniq_runner import BenchmarkResult, LabeledResult


def _run_tool_accuracy(
    tool: str, dataset_name: str, dataset_path: pathlib.Path, runs: int
) -> LabeledResult | None:
    """Accuracy path: dispatch to each runner's prediction function."""
    print(f"  [{tool}] dataset={dataset_name} runs={runs} (accuracy)")
    dispatch = {
        "spaniq": ("benchmarks.runners.spaniq_runner", "run_spaniq_predictions", ()),
        "groq": (
            "benchmarks.runners.groq_runner",
            "run_groq_predictions",
            (ImportError, EnvironmentError),
        ),
        "deepeval": (
            "benchmarks.runners.deepeval_runner",
            "run_deepeval_predictions",
            (ImportError, EnvironmentError),
        ),
        "ragas": (
            "benchmarks.runners.ragas_runner",
            "run_ragas_predictions",
            (ImportError, EnvironmentError, ValueError),
        ),
        "langfuse": (
            "benchmarks.runners.langfuse_runner",
            "run_langfuse_predictions",
            (ImportError, EnvironmentError),
        ),
    }
    if tool not in dispatch:
        print(f"    unknown tool: {tool}", file=sys.stderr)
        return None
    module_name, func_name, skip_excs = dispatch[tool]
    import importlib

    try:
        func = getattr(importlib.import_module(module_name), func_name)
        return func(dataset_path, n_runs=runs)
    except skip_excs as exc:
        print(f"    skipping {tool}: {exc}", file=sys.stderr)
        return None


def _run_tool(
    tool: str, dataset_name: str, dataset_path: pathlib.Path, runs: int
) -> BenchmarkResult | None:
    print(f"  [{tool}] dataset={dataset_name} runs={runs}")
    if tool == "spaniq":
        from benchmarks.runners.spaniq_runner import run_spaniq_eval

        return run_spaniq_eval(dataset_path, n_runs=runs)
    elif tool == "groq":
        try:
            from benchmarks.runners.groq_runner import run_groq_eval

            return run_groq_eval(dataset_path, n_runs=runs)
        except (OSError, ImportError) as exc:
            print(f"    skipping groq: {exc}", file=sys.stderr)
            return None
    elif tool == "deepeval":
        try:
            from benchmarks.runners.deepeval_runner import run_deepeval_eval

            return run_deepeval_eval(dataset_path, n_runs=runs)
        except (OSError, ImportError) as exc:
            print(f"    skipping deepeval: {exc}", file=sys.stderr)
            return None
    elif tool == "ragas":
        try:
            from benchmarks.runners.ragas_runner import run_ragas_eval

            return run_ragas_eval(dataset_path, n_runs=runs)
        except (OSError, ImportError, ValueError) as exc:
            print(f"    skipping ragas: {exc}", file=sys.stderr)
            return None
    elif tool == "langfuse":
        try:
            from benchmarks.runners.langfuse_runner import run_langfuse_eval

            return run_langfuse_eval(dataset_path, n_runs=runs)
        except (OSError, ImportError) as exc:
            print(f"    skipping langfuse: {exc}", file=sys.stderr)
            return None
    else:
        print(f"    unknown tool: {tool}", file=sys.stderr)
        return None


def main(
    tools: list[str] | None = None,
    datasets: list[str] | None = None,
    runs: int = 5,
    setup_only: bool = False,
    output_dir: str | pathlib.Path | None = None,
    metric: str = "accuracy",
) -> None:
    if setup_only:
        from benchmarks.datasets.fetch import main as fetch_main

        fetch_main()
        return

    tools = tools or ["spaniq"]
    datasets = datasets or ["all"]
    output_dir = pathlib.Path(output_dir) if output_dir else RESULTS_DIR

    dataset_names = list(DATASET_FILES.keys()) if "all" in datasets else datasets

    print(
        f"\nRunning benchmark: tools={tools} datasets={dataset_names} runs={runs} metric={metric}\n"
    )

    if metric == "accuracy":
        _run_accuracy(tools, dataset_names, runs, output_dir)
    else:
        _run_variance(tools, dataset_names, runs, output_dir)


def _run_accuracy(tools, dataset_names, runs, output_dir) -> None:
    results: list[LabeledResult] = []
    for dataset_name in dataset_names:
        path = DATASET_FILES.get(dataset_name)
        if path is None or not path.exists():
            print(f"  dataset not found: {dataset_name} — run: spaniq benchmark --setup")
            continue
        for tool in tools:
            r = _run_tool_accuracy(tool, dataset_name, path, runs)
            if r:
                results.append(r)

    if not results:
        print("No results to report.")
        return

    from benchmarks.analysis.report_accuracy import (
        build_reports,
        print_table,
        save_accuracy_csv,
        save_predictions_csv,
        save_summary_md,
    )

    reports = build_reports(results)
    print_table(reports)
    acc_path = save_accuracy_csv(reports, output_dir)
    pred_path = save_predictions_csv(results, output_dir)
    md_path = save_summary_md(reports, output_dir)

    print("\nResults saved:")
    print(f"  Accuracy CSV:  {acc_path}")
    print(f"  Predictions:   {pred_path}")
    print(f"  Summary:       {md_path}")


def _run_variance(tools, dataset_names, runs, output_dir) -> None:
    results: list[BenchmarkResult] = []
    for dataset_name in dataset_names:
        path = DATASET_FILES.get(dataset_name)
        if path is None or not path.exists():
            print(f"  dataset not found: {dataset_name} — run: spaniq benchmark --setup")
            continue
        for tool in tools:
            r = _run_tool(tool, dataset_name, path, runs)
            if r:
                results.append(r)

    if not results:
        print("No results to report.")
        return

    from benchmarks.analysis.report import (
        print_table,
        save_chart,
        save_csv,
        save_json,
        save_summary_md,
    )

    print_table(results)
    csv_path = save_csv(results, output_dir)
    json_path = save_json(results, output_dir)
    md_path = save_summary_md(results, output_dir)
    chart_path = save_chart(results, output_dir)

    print("\nResults saved:")
    print(f"  CSV:     {csv_path}")
    print(f"  JSON:    {json_path}")
    print(f"  Summary: {md_path}")
    if chart_path:
        print(f"  Chart:   {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="spanIQ accuracy benchmark suite: precision/recall/F1 at "
        "catching bad outputs (positive class = bad)"
    )
    parser.add_argument(
        "--tool",
        default="spaniq",
        help="comma-separated tools: spaniq,groq,deepeval,ragas,langfuse",
    )
    parser.add_argument(
        "--dataset", default="all", help="qa_factual, summarization, rag_retrieval, or all"
    )
    parser.add_argument("--runs", type=int, default=5, help="identical runs per tool (default: 5)")
    parser.add_argument(
        "--setup", action="store_true", help="download and cache benchmark datasets, then exit"
    )
    parser.add_argument("--output", default=str(RESULTS_DIR), help="directory to write results")
    parser.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "variance"],
        help="accuracy (precision/recall/F1, default) or legacy variance",
    )
    args = parser.parse_args()
    main(
        tools=[t.strip() for t in args.tool.split(",")],
        datasets=[d.strip() for d in args.dataset.split(",")],
        runs=args.runs,
        setup_only=args.setup,
        output_dir=args.output,
        metric=args.metric,
    )
