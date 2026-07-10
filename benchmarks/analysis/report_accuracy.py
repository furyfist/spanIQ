"""Accuracy-first reporting for the benchmark.

Consumes the per-tool LabeledResults, applies the calibration rule, and writes:

  - accuracy.csv          one row per tool+dataset: threshold, P, R, F1, AUC, AP
  - predictions.csv       every item: tool, dataset, label, score, decision
                          (the audit trail — rebuild any confusion matrix by hand)
  - accuracy_summary.md   the tables a reader actually reads

The determinism std dev is still reported, as a secondary column, honestly.
"""

from __future__ import annotations

import csv
import pathlib

from benchmarks.analysis.calibrate import AccuracyReport, evaluate_accuracy
from benchmarks.runners.spaniq_runner import LabeledResult


def build_reports(results: list[LabeledResult]) -> list[AccuracyReport]:
    return [evaluate_accuracy(r) for r in results]


def print_table(reports: list[AccuracyReport]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="spanIQ Accuracy Benchmark (positive class = bad)")
        for c in ["Tool", "Dataset", "Thr", "Precision", "Recall", "F1", "AUC", "AP", "Std"]:
            table.add_column(c, justify="left" if c in ("Tool", "Dataset") else "right")
        for r in reports:
            row = r.as_row()
            table.add_row(
                row["tool"],
                row["dataset"],
                f"{row['threshold']}",
                f"{row['precision']}",
                f"{row['recall']}",
                f"{row['f1']}",
                f"{row['roc_auc']}",
                f"{row['avg_precision']}",
                f"{row['score_std']}",
            )
        Console().print(table)
    except ImportError:
        for r in reports:
            row = r.as_row()
            print(
                f"{row['tool']:16s} {row['dataset']:14s} "
                f"P={row['precision']} R={row['recall']} F1={row['f1']} AUC={row['roc_auc']}"
            )


def save_accuracy_csv(reports: list[AccuracyReport], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "accuracy.csv"
    rows = [r.as_row() for r in reports]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_predictions_csv(results: list[LabeledResult], out_dir: pathlib.Path) -> pathlib.Path:
    """The audit trail: every item's mean score and its threshold-free decision
    inputs, so a skeptic can recompute precision/recall by hand."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tool", "dataset", "item_id", "true_label", "failure_kind", "mean_score"])
        for res in results:
            labels = res.true_labels
            scores = res.mean_scores
            kinds = [p.failure_kind for p in res.runs[0]] if res.runs else []
            for i, (lbl, sc) in enumerate(zip(labels, scores, strict=True)):
                writer.writerow(
                    [
                        res.tool,
                        res.dataset,
                        i,
                        lbl,
                        kinds[i] if i < len(kinds) else "",
                        round(sc, 6),
                    ]
                )
    return path


def save_summary_md(reports: list[AccuracyReport], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "accuracy_summary.md"
    lines = [
        "# spanIQ Accuracy Benchmark",
        "",
        "Positive class = **bad** (the failure we want to catch). Thresholds are "
        "chosen by F1 on a calibration fold; metrics below are on the held-out "
        "test fold. AUC and average precision are threshold-free.",
        "",
        "## Table 1 — Accuracy (test fold)",
        "",
        "| Tool | Dataset | Threshold | Precision | Recall | F1 | ROC-AUC | Avg Prec |",
        "|------|---------|-----------|-----------|--------|----|---------|----------|",
    ]
    for r in reports:
        row = r.as_row()
        lines.append(
            f"| {row['tool']} | {row['dataset']} | {row['threshold']} "
            f"| {row['precision']} | {row['recall']} | **{row['f1']}** "
            f"| {row['roc_auc']} | {row['avg_precision']} |"
        )
    lines += [
        "",
        "## Table 2 — Determinism sidebar (std dev of per-run mean score)",
        "",
        "Reported honestly: determinism is a property spanIQ has *by construction* "
        "(no LLM call), but LLM judges at `temperature=0` are often deterministic too.",
        "",
        "| Tool | Dataset | Score Std Dev |",
        "|------|---------|---------------|",
    ]
    for r in reports:
        row = r.as_row()
        lines.append(f"| {row['tool']} | {row['dataset']} | {row['score_std']} |")
    lines += [
        "",
        "> Recall (bad) is the money metric: of all genuinely bad outputs, how "
        "many did the tool catch? See `predictions.csv` to rebuild any number.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
