from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_DB = "spaniq.db"
# Groq retired llama-3.3-70b-versatile; pass --model for the model you run in production.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_METRICS = ["ResponseDriftMetric", "SemanticSimilarityMetric", "OutputStabilityMetric"]


class CLIError(Exception):
    """A user-facing error: printed as one line, exit status 1, no traceback."""


# ── helpers ───────────────────────────────────────────────────────────────────


def _metrics_from_names(spec: str) -> list:
    """Metric instances from a comma-separated list; accepts "ResponseDrift" or "ResponseDriftMetric"."""
    from spaniq.metrics.consistency import ConsistencyMetric
    from spaniq.metrics.output_stability import OutputStabilityMetric
    from spaniq.metrics.response_drift import ResponseDriftMetric
    from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

    factories = {
        cls.__name__: cls
        for cls in (
            ResponseDriftMetric,
            SemanticSimilarityMetric,
            OutputStabilityMetric,
            ConsistencyMetric,
        )
    }
    metrics = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        cls = factories.get(name) or factories.get(f"{name}Metric")
        if cls is None:
            valid = ", ".join(sorted(factories))
            raise CLIError(f"unknown metric '{name}' (valid: {valid})")
        metrics.append(cls())
    if not metrics:
        raise CLIError("--metrics is empty")
    return metrics


def _read_baseline_file(path: str) -> list[str]:
    """Outputs from a JSONL file of {"output": ...} objects, or plain text lines."""
    outputs: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                outputs.append(line)
                continue
            if isinstance(obj, dict):
                if "output" not in obj:
                    print(f"warning: line {lineno} has no 'output' field, skipped", file=sys.stderr)
                    continue
                obj = obj["output"]
            outputs.append(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))
    return outputs


def _baseline_collect(args: argparse.Namespace) -> None:
    from spaniq.monitor.baseline_store import BaselineStore

    store = BaselineStore(args.db)

    if args.source == "file":
        if not args.path:
            raise CLIError("--path is required when --source file")
        outputs = _read_baseline_file(args.path)
        if not outputs:
            raise CLIError(f"no outputs found in {args.path}")
        baseline_id = store.create(
            name=args.name,
            prompt=args.prompt,
            outputs=outputs,
            model_name=None,
        )
        print(f"created baseline '{args.name}' ({len(outputs)} outputs) — id: {baseline_id}")
        return

    # --source groq (default)
    try:
        from groq import Groq
    except ImportError as exc:
        raise CLIError("groq SDK not installed — run: pip install spaniq[groq]") from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise CLIError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)
    model = args.model or DEFAULT_GROQ_MODEL
    outputs = []
    rpm_delay = 60.0 / 28  # stay safely under 30 RPM

    print(f"collecting {args.n} outputs from {model} …")
    failure: Exception | None = None
    for i in range(args.n):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": args.prompt}],
            )
        except Exception as exc:
            failure = exc
            break
        outputs.append(resp.choices[0].message.content or "")
        print(f"  [{i + 1}/{args.n}] collected", end="\r")
        if i < args.n - 1:
            time.sleep(rpm_delay)

    print()
    if failure is not None:
        print(
            f"warning: Groq call failed after {len(outputs)} output(s): {failure}", file=sys.stderr
        )
        if not outputs:
            raise CLIError("no outputs collected, baseline not created")
        print("saving the outputs collected so far", file=sys.stderr)
    baseline_id = store.create(
        name=args.name,
        prompt=args.prompt,
        outputs=outputs,
        model_name=model,
    )
    print(f"created baseline '{args.name}' ({len(outputs)} outputs) — id: {baseline_id}")


def _baseline_list(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    from spaniq.monitor.baseline_store import BaselineStore

    store = BaselineStore(args.db)
    summaries = store.list_all()
    if not summaries:
        print("no baselines found")
        return

    console = Console()
    table = Table(title="Baselines")
    table.add_column("name", style="cyan")
    table.add_column("n_outputs", justify="right")
    table.add_column("version", justify="right")
    table.add_column("model")
    table.add_column("created_at")
    table.add_column("id", style="dim")

    for s in summaries:
        table.add_row(
            s.name,
            str(s.n_outputs),
            str(s.version),
            s.model_name or "—",
            s.created_at[:19],
            s.id,
        )
    console.print(table)


def _baseline_show(args: argparse.Namespace) -> None:
    import json

    from rich.console import Console

    from spaniq.monitor.baseline_store import BaselineStore

    store = BaselineStore(args.db)
    b = store.get_by_name(args.name)

    console = Console()
    console.print(f"[bold]{b.name}[/bold]  v{b.version}  ({b.n_outputs} outputs)")
    console.print(f"  id:          {b.id}")
    console.print(f"  prompt_hash: {b.prompt_hash}")
    console.print(f"  model:       {b.model_name or '—'}")
    console.print(f"  created:     {b.created_at}")
    console.print(f"  updated:     {b.updated_at}")
    console.print(f"\n[bold]prompt:[/bold] {b.prompt_text}\n")
    outputs = json.loads(b.outputs)
    console.print("[bold]sample outputs (first 3):[/bold]")
    for i, o in enumerate(outputs[:3], 1):
        console.print(f"  [{i}] {o[:120]}")


def _monitor_run(args: argparse.Namespace) -> None:
    from spaniq.monitor.collectors.file import FileCollector
    from spaniq.monitor.monitor import Monitor

    if args.source == "langfuse":
        import importlib.util

        if importlib.util.find_spec("langfuse") is None:
            raise CLIError("langfuse SDK not installed — run: pip install spaniq[langfuse]")
        from spaniq.monitor.collectors.langfuse import LangfuseCollector

        collector = LangfuseCollector(poll_interval=args.poll_interval)
    else:
        if not args.path:
            raise CLIError("--path is required when --source file")
        if not os.path.exists(args.path):
            raise CLIError(f"trace file not found: {args.path}")
        collector = FileCollector(args.path)

    metrics = _metrics_from_names(args.metrics) if args.metrics else None

    monitor = Monitor(
        baseline_name=args.baseline,
        collector=collector,
        metrics=metrics,
        db_path=args.db,
        alert_after=args.alert_after,
        alerts_path=args.alerts_path,
    )
    monitor.run(max_traces=args.max_traces)


def _pipeline_run(args: argparse.Namespace) -> None:
    from spaniq.attribution.pipeline_monitor import PipelineMonitor
    from spaniq.monitor.collectors.file import FileCollector

    collector = FileCollector(args.path)
    pm = PipelineMonitor(
        pipeline_name=args.name,
        collector=collector,
        db_path=args.db,
        warmup=args.warmup,
    )
    report = pm.run(max_traces=args.max_traces)
    if report.online_alarms:
        from rich.console import Console

        Console().print(f"[yellow]online alarms:[/yellow] {report.online_alarms}")


def _pipeline_status(args: argparse.Namespace) -> None:
    import numpy as np
    from rich.console import Console
    from rich.table import Table

    from spaniq.monitor.timeline_store import TimelineStore

    store = TimelineStore(args.db)
    comps = store.components(pipeline=args.name)
    if not comps:
        raise CLIError(f"no data for pipeline '{args.name}' — run 'spaniq pipeline run' first")
    metrics = [m.__class__.__name__ for m in _metrics_from_names(args.metrics)]
    console = Console()
    table = Table(title=f"Pipeline '{args.name}' — last {args.last} traces")
    table.add_column("component")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("trend", justify="right")
    table.add_column("pass_rate", justify="right")
    for comp in comps:
        for metric in metrics:
            rows = store.query(
                metric,
                last_n=args.last,
                component=comp,
                ascending=True,
                baseline_id=f"{args.name}/{comp}",
            )
            if not rows:
                continue
            scores = [r.score for r in rows]
            mean = float(np.mean(scores))
            trend = float(np.polyfit(range(len(scores)), scores, 1)[0]) if len(scores) >= 2 else 0.0
            pass_rate = sum(1 for r in rows if r.passed) / len(rows)
            table.add_row(
                comp, metric[:20], f"{mean:.4f}", f"{trend:+.6f}", f"{pass_rate * 100:.1f}%"
            )
    console.print(table)


def _attribute(args: argparse.Namespace) -> None:
    from spaniq.attribution.attributor import attribute
    from spaniq.attribution.report import (
        attribution_to_dict,
        print_attribution,
        save_attribution_png,
    )
    from spaniq.monitor.timeline_store import TimelineStore

    store = TimelineStore(args.db)
    components = store.components(pipeline=args.pipeline)
    if not components:
        known = ", ".join(store.pipelines()) or "none"
        raise CLIError(
            f"no data for pipeline '{args.pipeline}' (pipelines in {args.db}: {known}) — "
            "run 'spaniq pipeline run' first"
        )

    metrics = [m.__class__.__name__ for m in _metrics_from_names(args.metrics)]

    result = attribute(
        timeline=store,
        components=components,
        metrics=metrics,
        last_n=args.last,
        pelt_penalty=args.penalty,
        warmup=args.warmup,
        pipeline=args.pipeline,
    )

    if args.as_json:
        import json as _json

        print(_json.dumps(attribution_to_dict(result), indent=2))
    else:
        print_attribution(result)

    if args.export:
        save_attribution_png(
            result=result,
            timeline=store,
            components=components,
            primary_metric=metrics[0],
            path=args.export,
            last_n=args.last,
            pipeline=args.pipeline,
        )
        from rich.console import Console

        Console().print(f"[green]saved:[/green] {args.export}")


def _trace_to_json(trace) -> str:
    """A Trace as one FileCollector-compatible JSONL line."""
    record = {
        "trace_id": trace.trace_id,
        "timestamp": trace.timestamp,
        "input": trace.input,
        "output": trace.output,
    }
    if trace.metadata:
        record["metadata"] = trace.metadata
    if trace.components:
        record["components"] = [
            {
                "name": c.name,
                "kind": c.kind.value,
                "output": c.output,
                "latency_ms": c.latency_ms,
                "error": c.error,
            }
            for c in trace.components
        ]
    return json.dumps(record, ensure_ascii=False)


def _collect_otel(args: argparse.Namespace) -> None:
    if not args.store_only and not args.baseline:
        raise CLIError("--baseline is required unless --store-only is set")
    from spaniq.monitor.collectors.otel import OTelCollector

    collector = OTelCollector(
        grpc_port=args.grpc_port,
        http_port=args.http_port,
        assembly_timeout=args.assembly_timeout,
        host=args.host,
    )
    try:
        collector.start()
    except OSError as exc:
        raise CLIError(f"{exc} — pick free ports with --grpc-port / --http-port") from exc
    shown = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
    print(
        f"OTel collector listening on {args.host} — gRPC :{args.grpc_port}  HTTP :{args.http_port}"
    )
    print(
        f"Point your OTel exporter to http://{shown}:{args.grpc_port} (gRPC) "
        f"or http://{shown}:{args.http_port}/v1/traces (HTTP)"
    )
    print("Press Ctrl+C to stop.\n")

    if args.store_only:
        stored = 0
        try:
            with open(args.traces_path, "a", encoding="utf-8") as fh:
                for trace in collector.collect():
                    fh.write(_trace_to_json(trace) + "\n")
                    fh.flush()
                    stored += 1
                    print(f"  stored trace {trace.trace_id[:8]}… ({stored} total)")
        except KeyboardInterrupt:
            collector.stop()
            print(f"\nstopped. {stored} trace(s) appended to {args.traces_path}")
        return

    from spaniq.monitor.monitor import Monitor

    monitor = Monitor(
        baseline_name=args.baseline,
        collector=collector,
        db_path=args.db,
        alert_after=args.alert_after,
        alerts_path=args.alerts_path,
    )
    try:
        monitor.run()
    except KeyboardInterrupt:
        collector.stop()
        print("\nstopped.")


def _dashboard_launch(args: argparse.Namespace) -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("streamlit") is None:
            raise ImportError
    except ImportError:
        print(
            "error: streamlit not installed — run: pip install spaniq[dashboard]", file=sys.stderr
        )
        sys.exit(1)

    import pathlib

    app_path = pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--",
        "--db",
        args.db,
    ]
    print(f"Launching spanIQ dashboard at http://localhost:{args.port}")
    sys.exit(subprocess.call(cmd))


def _benchmark_run(args: argparse.Namespace) -> None:
    try:
        from benchmarks.run_benchmark import main as bench_main
    except ImportError as exc:
        # benchmarks/ (datasets, runners) is not part of the installed wheel
        raise CLIError(
            "the benchmark suite runs from a source checkout: "
            "git clone https://github.com/furyfist/spanIQ && cd spanIQ && "
            'pip install -e ".[benchmark]" && spaniq benchmark ...'
            f" ({exc})"
        ) from exc
    bench_main(
        tools=[t.strip() for t in args.tool.split(",")],
        datasets=[d.strip() for d in args.dataset.split(",")],
        runs=args.runs,
        setup_only=args.setup,
        output_dir=args.output,
        metric=args.metric,
    )


def _timeline_show(args: argparse.Namespace) -> None:
    from spaniq.monitor.timeline_store import TimelineStore
    from spaniq.monitor.visualize import print_sparkline

    store = TimelineStore(args.db)
    print_sparkline(store, args.metric, last_n=args.last)


def _timeline_export(args: argparse.Namespace) -> None:
    from rich.console import Console

    from spaniq.monitor.timeline_store import TimelineStore
    from spaniq.monitor.visualize import export_timeline_png

    store = TimelineStore(args.db)
    out = export_timeline_png(store, args.metric, output_path=args.output, last_n=args.last)
    Console().print(f"[green]saved:[/green] {out}")


def _timeline_summary(args: argparse.Namespace) -> None:
    from rich.console import Console

    from spaniq.monitor.timeline_store import TimelineStore

    store = TimelineStore(args.db)
    s = store.summary(args.metric, last_n=args.last)
    console = Console()
    if abs(s.trend) < 0.001:
        trend_label = "stable"
    elif s.trend > 0:
        trend_label = "worsening ↑"
    else:
        trend_label = "improving ↓"
    console.print(f"\n[bold]{s.metric_name}[/bold] (last {s.n} traces)")
    console.print(f"  mean:      {s.mean_score:.4f}")
    console.print(f"  std:       {s.std_score:.4f}")
    console.print(f"  min/max:   {s.min_score:.4f} / {s.max_score:.4f}")
    console.print(f"  pass rate: {s.pass_rate * 100:.1f}%")
    console.print(f"  trend:     {s.trend:+.6f}/trace ({trend_label})\n")


# ── parser ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spaniq")
    sub = parser.add_subparsers(dest="command")

    # --db is accepted before or after the subcommand ("spaniq baseline --db x list"
    # and "spaniq baseline list --db x"); SUPPRESS keeps the group value unless
    # the leaf flag is given.
    db_leaf = argparse.ArgumentParser(add_help=False)
    db_leaf.add_argument("--db", default=argparse.SUPPRESS, help="path to spaniq.db")

    # ── test run ──────────────────────────────────────────────────────────────
    test_p = sub.add_parser("test", help="run tests")
    test_sub = test_p.add_subparsers(dest="test_command")
    run_p = test_sub.add_parser("run", help="run pytest")
    run_p.add_argument("pytest_args", nargs=argparse.REMAINDER)

    # ── baseline ──────────────────────────────────────────────────────────────
    bl_p = sub.add_parser("baseline", help="manage baselines")
    bl_p.add_argument("--db", default=DEFAULT_DB, help="path to spaniq.db")
    bl_sub = bl_p.add_subparsers(dest="baseline_command")

    col_p = bl_sub.add_parser("collect", help="collect baseline outputs", parents=[db_leaf])
    col_p.add_argument("--name", required=True, help="baseline name")
    col_p.add_argument("--prompt", required=True, help="prompt text")
    col_p.add_argument("--source", choices=["groq", "file"], default="groq")
    col_p.add_argument("--model", default=None, help="model name (groq source)")
    col_p.add_argument("--n", type=int, default=50, help="number of outputs (groq source)")
    col_p.add_argument("--path", default=None, help="JSONL file path (file source)")

    bl_sub.add_parser("list", help="list all baselines", parents=[db_leaf])

    show_p = bl_sub.add_parser("show", help="inspect a baseline", parents=[db_leaf])
    show_p.add_argument("name", help="baseline name")

    # ── monitor ───────────────────────────────────────────────────────────────
    mon_p = sub.add_parser("monitor", help="run production monitoring")
    mon_p.add_argument("--db", default=DEFAULT_DB)
    mon_sub = mon_p.add_subparsers(dest="monitor_command")

    run_mon = mon_sub.add_parser("run", help="run monitor loop", parents=[db_leaf])
    run_mon.add_argument("--baseline", required=True, help="baseline name")
    run_mon.add_argument("--source", choices=["file", "langfuse"], default="file")
    run_mon.add_argument("--path", default=None, help="JSONL file path (file source)")
    run_mon.add_argument("--poll-interval", type=int, default=30, dest="poll_interval")
    run_mon.add_argument(
        "--metrics",
        default=None,
        help="comma-separated metric names, e.g. ResponseDrift,SemanticSimilarity "
        "(default: ResponseDrift, SemanticSimilarity, OutputStability)",
    )
    run_mon.add_argument("--alert-after", type=int, default=3, dest="alert_after")
    run_mon.add_argument("--alerts-path", default="alerts.jsonl", dest="alerts_path")
    run_mon.add_argument("--max-traces", type=int, default=None, dest="max_traces")

    # ── timeline ──────────────────────────────────────────────────────────────
    tl_p = sub.add_parser("timeline", help="inspect the score timeline")
    tl_p.add_argument("--db", default=DEFAULT_DB)
    tl_sub = tl_p.add_subparsers(dest="timeline_command")

    show_tl = tl_sub.add_parser("show", help="print sparkline to terminal", parents=[db_leaf])
    show_tl.add_argument("--metric", required=True)
    show_tl.add_argument("--last", type=int, default=50)

    export_tl = tl_sub.add_parser("export", help="export PNG chart", parents=[db_leaf])
    export_tl.add_argument("--metric", required=True)
    export_tl.add_argument("--last", type=int, default=200)
    export_tl.add_argument("--output", default="timeline.png")

    summary_tl = tl_sub.add_parser("summary", help="print aggregate stats", parents=[db_leaf])
    summary_tl.add_argument("--metric", required=True)
    summary_tl.add_argument("--last", type=int, default=200)

    # ── pipeline ──────────────────────────────────────────────────────────────
    pip_p = sub.add_parser("pipeline", help="run per-component pipeline monitoring (V3)")
    pip_p.add_argument("--db", default=DEFAULT_DB)
    pip_sub = pip_p.add_subparsers(dest="pipeline_command")

    pip_run = pip_sub.add_parser(
        "run", help="run PipelineMonitor on a JSONL trace file", parents=[db_leaf]
    )
    pip_run.add_argument("--name", required=True, help="pipeline name")
    pip_run.add_argument("--path", required=True, help="JSONL trace file with components")
    pip_run.add_argument("--max-traces", type=int, default=None, dest="max_traces")
    pip_run.add_argument("--warmup", type=int, default=20)

    pip_status = pip_sub.add_parser(
        "status", help="show score statistics per component of a pipeline", parents=[db_leaf]
    )
    pip_status.add_argument("--name", required=True, help="pipeline name")
    pip_status.add_argument("--last", type=int, default=200)
    pip_status.add_argument("--metrics", default=",".join(DEFAULT_METRICS))

    # ── attribute ─────────────────────────────────────────────────────────────
    attr_p = sub.add_parser("attribute", help="run failure attribution on stored timeline (V3)")
    attr_p.add_argument("--db", default=DEFAULT_DB)
    attr_p.add_argument("--pipeline", required=True, help="pipeline name")
    attr_p.add_argument("--last", type=int, default=500)
    attr_p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    attr_p.add_argument("--json", action="store_true", dest="as_json", help="output JSON")
    attr_p.add_argument("--export", default=None, help="save PNG chart to this path")
    attr_p.add_argument(
        "--penalty", type=float, default=None, help="PELT penalty (default: BIC = 3*log(n))"
    )
    attr_p.add_argument(
        "--warmup", type=int, default=20, help="warmup traces to add back to PELT indices"
    )

    # ── collect-otel ──────────────────────────────────────────────────────────
    otel_p = sub.add_parser("collect-otel", help="receive OTel spans via OTLP and run monitoring")
    otel_p.add_argument(
        "--baseline", default=None, help="baseline name to monitor against (not with --store-only)"
    )
    otel_p.add_argument("--db", default=DEFAULT_DB)
    otel_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind; 0.0.0.0 accepts other machines (no TLS or auth)",
    )
    otel_p.add_argument("--grpc-port", type=int, default=4317, dest="grpc_port")
    otel_p.add_argument("--http-port", type=int, default=4318, dest="http_port")
    otel_p.add_argument(
        "--store-only",
        action="store_true",
        dest="store_only",
        help="append received traces to --traces-path as JSONL instead of monitoring; "
        "replay later with 'spaniq monitor run --path' or 'spaniq pipeline run --path'",
    )
    otel_p.add_argument(
        "--traces-path",
        default="otel_traces.jsonl",
        dest="traces_path",
        help="JSONL output for --store-only",
    )
    otel_p.add_argument("--assembly-timeout", type=float, default=5.0, dest="assembly_timeout")
    otel_p.add_argument("--alert-after", type=int, default=3, dest="alert_after")
    otel_p.add_argument("--alerts-path", default="alerts.jsonl", dest="alerts_path")

    # ── dashboard ─────────────────────────────────────────────────────────────
    dash_p = sub.add_parser("dashboard", help="launch the Streamlit local dashboard")
    dash_p.add_argument("--db", default=DEFAULT_DB, help="path to spaniq.db")
    dash_p.add_argument("--port", type=int, default=8501, help="Streamlit port")

    # ── benchmark ─────────────────────────────────────────────────────────────
    bench_p = sub.add_parser("benchmark", help="run the accuracy benchmark suite")
    bench_p.add_argument(
        "--tool",
        default="spaniq",
        help="comma-separated tools to benchmark: spaniq,deepeval,ragas,groq,langfuse",
    )
    bench_p.add_argument(
        "--dataset",
        default="all",
        help="dataset to use: qa_factual, summarization, rag_retrieval, all",
    )
    bench_p.add_argument("--runs", type=int, default=5, help="number of identical runs per tool")
    bench_p.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "variance"],
        help="accuracy (precision/recall/F1, default) or legacy variance",
    )
    bench_p.add_argument(
        "--setup", action="store_true", help="download and cache benchmark datasets, then exit"
    )
    bench_p.add_argument(
        "--output", default="benchmarks/results", help="directory to write results"
    )

    # ── demo ──────────────────────────────────────────────────────────────────
    demo_p = sub.add_parser("demo", help="run reproducible replay demos")
    demo_sub = demo_p.add_subparsers(dest="demo_command")

    for name in ["prompt-injection", "model-swap", "rag-breakage", "run-all"]:
        dp = demo_sub.add_parser(name, help=f"run {name} demo")
        dp.add_argument("--offline", action="store_true", help="use pre-generated fixtures")

    return parser


# ── entrypoint ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _dispatch(parser, args)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyError as exc:
        # BaselineStore raises KeyError("Baseline not found: <name>")
        message = exc.args[0] if exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "test":
        if args.test_command == "run":
            sys.exit(subprocess.call([sys.executable, "-m", "pytest", *args.pytest_args, "-v"]))
        else:
            parser.parse_args(["test", "--help"])

    elif args.command == "baseline":
        if args.baseline_command == "collect":
            _baseline_collect(args)
        elif args.baseline_command == "list":
            _baseline_list(args)
        elif args.baseline_command == "show":
            _baseline_show(args)
        else:
            parser.parse_args(["baseline", "--help"])

    elif args.command == "monitor":
        if args.monitor_command == "run":
            _monitor_run(args)
        else:
            parser.parse_args(["monitor", "--help"])

    elif args.command == "pipeline":
        if args.pipeline_command == "run":
            _pipeline_run(args)
        elif args.pipeline_command == "status":
            _pipeline_status(args)
        else:
            parser.parse_args(["pipeline", "--help"])

    elif args.command == "attribute":
        _attribute(args)

    elif args.command == "timeline":
        if args.timeline_command == "show":
            _timeline_show(args)
        elif args.timeline_command == "export":
            _timeline_export(args)
        elif args.timeline_command == "summary":
            _timeline_summary(args)
        else:
            parser.parse_args(["timeline", "--help"])

    elif args.command == "collect-otel":
        _collect_otel(args)

    elif args.command == "dashboard":
        _dashboard_launch(args)

    elif args.command == "benchmark":
        _benchmark_run(args)

    elif args.command == "demo":
        offline = getattr(args, "offline", False)
        if args.demo_command == "prompt-injection":
            from spaniq.demos.prompt_injection import run

            run(offline=offline)
        elif args.demo_command == "model-swap":
            from spaniq.demos.model_swap import run

            run(offline=offline)
        elif args.demo_command == "rag-breakage":
            from spaniq.demos.rag_breakage import run

            run(offline=offline)
        elif args.demo_command == "run-all":
            from spaniq.demos.model_swap import run as run_ms
            from spaniq.demos.prompt_injection import run as run_pi
            from spaniq.demos.rag_breakage import run as run_rag

            run_pi(offline=offline)
            run_ms(offline=offline)
            run_rag(offline=offline)
        else:
            parser.parse_args(["demo", "--help"])

    else:
        parser.print_help()
