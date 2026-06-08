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
