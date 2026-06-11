"""V3 cascade attribution demo.

Shows a 3-component RAG pipeline where retrieval breaks at trace ~101,
generation degrades ~7 traces later, and search_tool stays healthy.
spanIQ V3 localizes both breaks and identifies retrieval as root cause.

Usage:
    python -m spaniq.demos.cascade_pipeline          # offline fixtures
    python -m spaniq.demos.cascade_pipeline --live   # regenerate via Groq (requires GROQ_API_KEY)
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cascade" / "traces.jsonl"
BREAK_TRACE = 101
GENERATION_LAG = 7


def run(offline: bool = True) -> None:
    if not offline:
        console.print("[yellow]--live mode not implemented in this version; using fixtures[/yellow]")

    if not FIXTURE_PATH.exists():
        from spaniq.demos._gen_cascade_fixtures import generate
        generate()

    console.rule("[bold]spanIQ V3 — Cascade Attribution Demo[/bold]")
    console.print(
        f"Traces 1–{BREAK_TRACE}: all healthy\n"
        f"Trace {BREAK_TRACE}+: retrieval returns empty/garbage context\n"
        f"Trace {BREAK_TRACE + GENERATION_LAG}+: generation degrades (cascade)\n"
        f"search_tool: healthy throughout (control)\n"
    )

    from spaniq.attribution.attributor import attribute
    from spaniq.attribution.pipeline_monitor import PipelineMonitor
    from spaniq.attribution.report import print_attribution, save_attribution_png
    from spaniq.monitor.collectors.file import FileCollector

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    collector = FileCollector(str(FIXTURE_PATH))
    pm = PipelineMonitor(
        pipeline_name="cascade-demo",
        collector=collector,
        db_path=db_path,
        window_size=20,
        warmup=20,
    )

    console.print("[dim]running PipelineMonitor...[/dim]")
    report = pm.run()
    console.print(f"processed {report.total_traces} traces, components: {report.components_seen}")

    if report.online_alarms:
        console.print(f"[yellow]online CUSUM alarms:[/yellow] {report.online_alarms}")

    from spaniq.monitor.timeline_store import TimelineStore
    store = TimelineStore(db_path)
    components = store.components()

    metrics = ["ResponseDriftMetric", "SemanticSimilarityMetric", "OutputStabilityMetric"]
    result = attribute(
        timeline=store,
        components=components,
        metrics=metrics,
        last_n=200,
        cusum_alarms=report.online_alarms,
    )

    console.rule("[bold]Attribution Result[/bold]")
    print_attribution(result)

    png_path = Path(__file__).parent / "fixtures" / "cascade" / "attribution.png"
    try:
        save_attribution_png(
            result=result,
            timeline=store,
            components=components,
            primary_metric="ResponseDriftMetric",
            path=str(png_path),
            last_n=200,
        )
        console.print(f"[green]chart saved:[/green] {png_path}")
    except Exception as e:
        console.print(f"[dim]chart not saved: {e}[/dim]")

    import os
    try:
        os.unlink(db_path)
    except OSError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cascade attribution demo")
    parser.add_argument("--live", action="store_true", help="regenerate via Groq")
    args = parser.parse_args()
    run(offline=not args.live)
