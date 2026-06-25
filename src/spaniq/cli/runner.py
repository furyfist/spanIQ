from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# ── helpers ───────────────────────────────────────────────────────────────────


def _baseline_collect(args: argparse.Namespace) -> None:
    from spaniq.monitor.baseline_store import BaselineStore

    store = BaselineStore(args.db)

    if args.source == "file":
        if not args.path:
            print("error: --path is required when --source file", file=sys.stderr)
            sys.exit(1)
        outputs: list[str] = []
        with open(args.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    outputs.append(obj.get("output", obj) if isinstance(obj, dict) else str(obj))
                except json.JSONDecodeError:
                    outputs.append(line)
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
    except ImportError:
        print("error: groq SDK not installed — run: pip install spaniq[groq]", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("error: GROQ_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)
    model = args.model or "llama-3.3-70b-versatile"
    outputs = []
    rpm_delay = 60.0 / 28  # stay safely under 30 RPM

    print(f"collecting {args.n} outputs from {model} …")
    for i in range(args.n):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
        )
        outputs.append(resp.choices[0].message.content or "")
        print(f"  [{i + 1}/{args.n}] collected", end="\r")
        if i < args.n - 1:
            time.sleep(rpm_delay)

    print()
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
    try:
        b = store.get_by_name(args.name)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

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
        try:
            from spaniq.monitor.collectors.langfuse import LangfuseCollector
        except ImportError:
            print(
                "error: langfuse SDK not installed — run: pip install spaniq[langfuse]",
                file=sys.stderr,
            )
            sys.exit(1)
        collector = LangfuseCollector(poll_interval=args.poll_interval)
    elif args.source == "file":
        if not args.path:
            print("error: --path is required when --source file", file=sys.stderr)
            sys.exit(1)
        collector = FileCollector(args.path)
    else:
        print(f"error: unknown source '{args.source}'", file=sys.stderr)
        sys.exit(1)

    metrics = None
    if args.metrics:
        from spaniq.metrics.consistency import ConsistencyMetric
        from spaniq.metrics.output_stability import OutputStabilityMetric
        from spaniq.metrics.response_drift import ResponseDriftMetric
        from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric

        metric_map = {
            "ResponseDrift": ResponseDriftMetric(),
            "SemanticSimilarity": SemanticSimilarityMetric(),
            "OutputStability": OutputStabilityMetric(),
            "Consistency": ConsistencyMetric(),
        }
        names = [n.strip() for n in args.metrics.split(",")]
        metrics = [metric_map[n] for n in names if n in metric_map]

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
        window_size=args.window,
        warmup=args.warmup,
    )
    report = pm.run(max_traces=args.max_traces)
    if report.online_alarms:
        from rich.console import Console
        Console().print(f"[yellow]online alarms:[/yellow] {report.online_alarms}")


def _pipeline_status(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    from spaniq.monitor.timeline_store import TimelineStore

    store = TimelineStore(args.db)
    comps = store.components()
    console = Console()
    table = Table(title=f"Pipeline components — last {args.last} traces")
    table.add_column("component")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("trend", justify="right")
    table.add_column("pass_rate", justify="right")
    for comp in comps:
        for metric in ["ResponseDriftMetric", "SemanticSimilarityMetric", "OutputStabilityMetric"]:
            rows = store.query(metric, last_n=args.last, component=comp, ascending=True)
            if not rows:
                continue
            import numpy as np
            scores = [r.score for r in rows]
            mean = float(np.mean(scores))
            trend = float(np.polyfit(range(len(scores)), scores, 1)[0]) if len(scores) >= 2 else 0.0
            pass_rate = sum(1 for r in rows if r.passed) / len(rows)
            table.add_row(comp, metric[:20], f"{mean:.4f}", f"{trend:+.6f}", f"{pass_rate*100:.1f}%")
    console.print(table)


def _attribute(args: argparse.Namespace) -> None:
    from spaniq.attribution.attributor import attribute
    from spaniq.attribution.report import (
        attribution_to_dict,
        print_attribution,
        save_attribution_json,
        save_attribution_png,
    )
    from spaniq.monitor.timeline_store import TimelineStore

    store = TimelineStore(args.db)
    components = store.components()
    if not components:
        print("no component data found in timeline — run 'spaniq pipeline run' first")
        return

    metrics = (
        [m.strip() for m in args.metrics.split(",")]
        if args.metrics
        else ["ResponseDriftMetric", "SemanticSimilarityMetric", "OutputStabilityMetric"]
    )

    result = attribute(
        timeline=store,
        components=components,
        metrics=metrics,
        last_n=args.last,
        pelt_penalty=args.penalty,
        warmup=args.warmup,
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
        )
        from rich.console import Console
        Console().print(f"[green]saved:[/green] {args.export}")


def _collect_otel(args: argparse.Namespace) -> None:
    try:
        from spaniq.monitor.collectors.otel import OTelCollector
    except ImportError:
        print("error: OTel deps not installed — run: pip install spaniq[otel]", file=sys.stderr)
        sys.exit(1)

    collector = OTelCollector(
        grpc_port=args.grpc_port,
        http_port=args.http_port,
        assembly_timeout=args.assembly_timeout,
    )
    collector.start()
    print(f"OTel collector listening — gRPC :{args.grpc_port}  HTTP :{args.http_port}")
    print("Point your OTel exporter to localhost:4317 (gRPC) or localhost:4318 (HTTP)")
    print("Press Ctrl+C to stop.\n")

    if args.store_only:
        from spaniq.monitor.timeline_store import TimelineStore
        store = TimelineStore(args.db)
        try:
            for trace in collector.collect():
                print(f"  trace received: {trace.trace_id[:8]}…")
        except KeyboardInterrupt:
            collector.stop()
            print("\nstopped.")
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
        print("error: streamlit not installed — run: pip install spaniq[dashboard]", file=sys.stderr)
        sys.exit(1)

    import pathlib
    app_path = pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--",
        "--db", args.db,
    ]
    print(f"Launching spanIQ dashboard at http://localhost:{args.port}")
    sys.exit(subprocess.call(cmd))


def _benchmark_run(args: argparse.Namespace) -> None:
    try:
        from benchmarks.run_benchmark import main as bench_main
    except ImportError as exc:
        print(f"error: benchmark deps not available — {exc}", file=sys.stderr)
        sys.exit(1)
    bench_main(
        tools=[t.strip() for t in args.tool.split(",")],
        datasets=[d.strip() for d in args.dataset.split(",")],
        runs=args.runs,
        setup_only=args.setup,
        output_dir=args.output,
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

    # ── test run ──────────────────────────────────────────────────────────────
    test_p = sub.add_parser("test", help="run tests")
    test_sub = test_p.add_subparsers(dest="test_command")
    run_p = test_sub.add_parser("run", help="run pytest")
    run_p.add_argument("pytest_args", nargs=argparse.REMAINDER)

    # ── baseline ──────────────────────────────────────────────────────────────
    bl_p = sub.add_parser("baseline", help="manage baselines")
    bl_p.add_argument("--db", default="spaniq.db", help="path to spaniq.db")
    bl_sub = bl_p.add_subparsers(dest="baseline_command")

    col_p = bl_sub.add_parser("collect", help="collect baseline outputs")
    col_p.add_argument("--name", required=True, help="baseline name")
    col_p.add_argument("--prompt", required=True, help="prompt text")
    col_p.add_argument("--source", choices=["groq", "file"], default="groq")
    col_p.add_argument("--model", default=None, help="model name (groq source)")
    col_p.add_argument("--n", type=int, default=50, help="number of outputs (groq source)")
    col_p.add_argument("--path", default=None, help="JSONL file path (file source)")

    bl_sub.add_parser("list", help="list all baselines")

    show_p = bl_sub.add_parser("show", help="inspect a baseline")
    show_p.add_argument("name", help="baseline name")

    # ── monitor ───────────────────────────────────────────────────────────────
    mon_p = sub.add_parser("monitor", help="run production monitoring")
    mon_p.add_argument("--db", default="spaniq.db")
    mon_sub = mon_p.add_subparsers(dest="monitor_command")

    run_mon = mon_sub.add_parser("run", help="run monitor loop")
    run_mon.add_argument("--baseline", required=True, help="baseline name")
    run_mon.add_argument("--source", choices=["file", "langfuse"], default="file")
    run_mon.add_argument("--path", default=None, help="JSONL file path (file source)")
    run_mon.add_argument("--poll-interval", type=int, default=30, dest="poll_interval")
    run_mon.add_argument("--metrics", default=None, help="comma-separated metric names")
    run_mon.add_argument("--alert-after", type=int, default=3, dest="alert_after")
    run_mon.add_argument("--alerts-path", default="alerts.jsonl", dest="alerts_path")
    run_mon.add_argument("--max-traces", type=int, default=None, dest="max_traces")

    # ── timeline ──────────────────────────────────────────────────────────────
    tl_p = sub.add_parser("timeline", help="inspect the score timeline")
    tl_p.add_argument("--db", default="spaniq.db")
    tl_sub = tl_p.add_subparsers(dest="timeline_command")

    show_tl = tl_sub.add_parser("show", help="print sparkline to terminal")
    show_tl.add_argument("--metric", required=True)
    show_tl.add_argument("--last", type=int, default=50)

    export_tl = tl_sub.add_parser("export", help="export PNG chart")
    export_tl.add_argument("--metric", required=True)
    export_tl.add_argument("--last", type=int, default=200)
    export_tl.add_argument("--output", default="timeline.png")

    summary_tl = tl_sub.add_parser("summary", help="print aggregate stats")
    summary_tl.add_argument("--metric", required=True)
    summary_tl.add_argument("--last", type=int, default=200)

    # ── pipeline ──────────────────────────────────────────────────────────────
    pip_p = sub.add_parser("pipeline", help="run per-component pipeline monitoring (V3)")
    pip_p.add_argument("--db", default="spaniq.db")
    pip_sub = pip_p.add_subparsers(dest="pipeline_command")

    pip_run = pip_sub.add_parser("run", help="run PipelineMonitor on a JSONL trace file")
    pip_run.add_argument("--name", required=True, help="pipeline name")
    pip_run.add_argument("--path", required=True, help="JSONL trace file with components")
    pip_run.add_argument("--max-traces", type=int, default=None, dest="max_traces")
    pip_run.add_argument("--window", type=int, default=20)
    pip_run.add_argument("--warmup", type=int, default=20)

    pip_status = pip_sub.add_parser("status", help="show live CUSUM statistics per component")
    pip_status.add_argument("--name", required=True, help="pipeline name")
    pip_status.add_argument("--last", type=int, default=200)

    # ── attribute ─────────────────────────────────────────────────────────────
    attr_p = sub.add_parser("attribute", help="run failure attribution on stored timeline (V3)")
    attr_p.add_argument("--db", default="spaniq.db")
    attr_p.add_argument("--pipeline", required=True, help="pipeline name")
    attr_p.add_argument("--last", type=int, default=500)
    attr_p.add_argument("--metrics", default=None, help="comma-separated metric names")
    attr_p.add_argument("--json", action="store_true", dest="as_json", help="output JSON")
    attr_p.add_argument("--export", default=None, help="save PNG chart to this path")
    attr_p.add_argument("--penalty", type=float, default=None, help="PELT penalty (default: BIC = 3*log(n))")
    attr_p.add_argument("--warmup", type=int, default=20, help="warmup traces to add back to PELT indices")

    # ── collect-otel ──────────────────────────────────────────────────────────
    otel_p = sub.add_parser("collect-otel", help="receive OTel spans via OTLP and run monitoring")
    otel_p.add_argument("--baseline", required=True, help="baseline name to monitor against")
    otel_p.add_argument("--db", default="spaniq.db")
    otel_p.add_argument("--grpc-port", type=int, default=4317, dest="grpc_port")
    otel_p.add_argument("--http-port", type=int, default=4318, dest="http_port")
    otel_p.add_argument("--store-only", action="store_true", dest="store_only",
                        help="collect and store traces without running metric monitoring")
    otel_p.add_argument("--assembly-timeout", type=float, default=5.0, dest="assembly_timeout")
    otel_p.add_argument("--alert-after", type=int, default=3, dest="alert_after")
    otel_p.add_argument("--alerts-path", default="alerts.jsonl", dest="alerts_path")

    # ── dashboard ─────────────────────────────────────────────────────────────
    dash_p = sub.add_parser("dashboard", help="launch the Streamlit local dashboard")
    dash_p.add_argument("--db", default="spaniq.db", help="path to spaniq.db")
    dash_p.add_argument("--port", type=int, default=8501, help="Streamlit port")

    # ── benchmark ─────────────────────────────────────────────────────────────
    bench_p = sub.add_parser("benchmark", help="run the determinism benchmark suite")
    bench_p.add_argument("--tool", default="spaniq",
                         help="comma-separated tools to benchmark: spaniq,deepeval,ragas,groq,langfuse")
    bench_p.add_argument("--dataset", default="all",
                         help="dataset to use: qa_factual, summarization, rag_retrieval, all")
    bench_p.add_argument("--runs", type=int, default=5, help="number of identical runs per tool")
    bench_p.add_argument("--setup", action="store_true",
                         help="download and cache benchmark datasets, then exit")
    bench_p.add_argument("--output", default="benchmarks/results",
                         help="directory to write results")

    # ── demo ──────────────────────────────────────────────────────────────────
    demo_p = sub.add_parser("demo", help="run reproducible replay demos")
    demo_sub = demo_p.add_subparsers(dest="demo_command")

    for name in ["prompt-injection", "model-swap", "rag-breakage", "run-all"]:
        dp = demo_sub.add_parser(name, help=f"run {name} demo")
        dp.add_argument("--offline", action="store_true", help="use pre-generated fixtures")

    return parser


# ── entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "test":
        if args.test_command == "run":
            sys.exit(subprocess.call([sys.executable, "-m", "pytest", *args.pytest_args, "-v"]))
        else:
            parser.parse_args(["test", "--help"])

    elif args.command == "baseline":
        if not hasattr(args, "db"):
            args.db = "spaniq.db"
        if args.baseline_command == "collect":
            _baseline_collect(args)
        elif args.baseline_command == "list":
            _baseline_list(args)
        elif args.baseline_command == "show":
            _baseline_show(args)
        else:
            parser.parse_args(["baseline", "--help"])

    elif args.command == "monitor":
        if not hasattr(args, "db"):
            args.db = "spaniq.db"
        if args.monitor_command == "run":
            _monitor_run(args)
        else:
            parser.parse_args(["monitor", "--help"])

    elif args.command == "pipeline":
        if not hasattr(args, "db"):
            args.db = "spaniq.db"
        if args.pipeline_command == "run":
            _pipeline_run(args)
        elif args.pipeline_command == "status":
            _pipeline_status(args)
        else:
            parser.parse_args(["pipeline", "--help"])

    elif args.command == "attribute":
        if not hasattr(args, "db"):
            args.db = "spaniq.db"
        _attribute(args)

    elif args.command == "timeline":
        if not hasattr(args, "db"):
            args.db = "spaniq.db"
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
