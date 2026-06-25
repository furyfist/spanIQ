"""CLI entry point for the determinism benchmark suite.

Usage (via spaniq CLI):
    spaniq benchmark --tool spaniq --runs 5
    spaniq benchmark --tool spaniq,groq --runs 3 --dataset qa_factual
    spaniq benchmark --setup   # download full HF datasets

Direct:
    python -m benchmarks.run_benchmark --tool spaniq --runs 5
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from benchmarks.config import DATASET_FILES, RESULTS_DIR
from benchmarks.runners.spaniq_runner import BenchmarkResult


def _run_tool(tool: str, dataset_name: str, dataset_path: pathlib.Path,
              runs: int) -> BenchmarkResult | None:
    print(f"  [{tool}] dataset={dataset_name} runs={runs}")
    if tool == "spaniq":
        from benchmarks.runners.spaniq_runner import run_spaniq_eval
        return run_spaniq_eval(dataset_path, n_runs=runs)
    elif tool == "groq":
        try:
            from benchmarks.runners.groq_runner import run_groq_eval
            return run_groq_eval(dataset_path, n_runs=runs)
        except (ImportError, EnvironmentError) as exc:
            print(f"    skipping groq: {exc}", file=sys.stderr)
            return None
    elif tool == "deepeval":
        try:
            from benchmarks.runners.deepeval_runner import run_deepeval_eval
            return run_deepeval_eval(dataset_path, n_runs=runs)
        except (ImportError, EnvironmentError) as exc:
            print(f"    skipping deepeval: {exc}", file=sys.stderr)
            return None
    elif tool == "ragas":
        try:
            from benchmarks.runners.ragas_runner import run_ragas_eval
            return run_ragas_eval(dataset_path, n_runs=runs)
        except (ImportError, EnvironmentError, ValueError) as exc:
            print(f"    skipping ragas: {exc}", file=sys.stderr)
            return None
    elif tool == "langfuse":
        try:
            from benchmarks.runners.langfuse_runner import run_langfuse_eval
            return run_langfuse_eval(dataset_path, n_runs=runs)
        except (ImportError, EnvironmentError) as exc:
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
) -> None:
    if setup_only:
        from benchmarks.datasets.fetch import main as fetch_main
        fetch_main()
        return

    tools = tools or ["spaniq"]
    datasets = datasets or ["all"]
    output_dir = pathlib.Path(output_dir) if output_dir else RESULTS_DIR

    dataset_names: list[str]
    if "all" in datasets:
        dataset_names = list(DATASET_FILES.keys())
    else:
        dataset_names = datasets

    results: list[BenchmarkResult] = []
    print(f"\nRunning benchmark: tools={tools} datasets={dataset_names} runs={runs}\n")

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
        print_table, save_csv, save_json, save_chart, save_summary_md,
    )
    print_table(results)
    csv_path = save_csv(results, output_dir)
    json_path = save_json(results, output_dir)
    md_path = save_summary_md(results, output_dir)
    chart_path = save_chart(results, output_dir)

    print(f"\nResults saved:")
    print(f"  CSV:     {csv_path}")
    print(f"  JSON:    {json_path}")
    print(f"  Summary: {md_path}")
    if chart_path:
        print(f"  Chart:   {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spanIQ determinism benchmark suite")
    parser.add_argument("--tool", default="spaniq")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--output", default=str(RESULTS_DIR))
    args = parser.parse_args()
    main(
        tools=[t.strip() for t in args.tool.split(",")],
        datasets=[d.strip() for d in args.dataset.split(",")],
        runs=args.runs,
        setup_only=args.setup,
        output_dir=args.output,
    )
