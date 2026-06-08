"""Demo 2: Model Swap Detection

Baseline: 20 outputs from llama-3.3-70b-versatile on coding questions.
Normal traces: 20 more from 70b.
Swapped traces: 20 from llama-3.1-8b-instant (cheaper, shorter outputs).

Expected: OutputStabilityMetric detects structural change (word count drops).
ResponseDrift may also fire.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "model_swap"
OUTPUT_DIR = Path(__file__).parent / "output"

PROMPT = "Write a short Python function that reverses a string and explain how it works."
SYSTEM = "You are a helpful coding assistant. Give clear, concise answers."
MODEL_70B = "llama-3.3-70b-versatile"
MODEL_8B = "llama-3.1-8b-instant"


def run(offline: bool = False, db_path: str = "spaniq_demo_swap.db") -> None:
    from rich.console import Console

    from spaniq.metrics.output_stability import OutputStabilityMetric
    from spaniq.metrics.response_drift import ResponseDriftMetric
    from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
    from spaniq.monitor.baseline_store import BaselineStore
    from spaniq.monitor.collectors.file import FileCollector
    from spaniq.monitor.monitor import Monitor
    from spaniq.monitor.visualize import export_timeline_png

    console = Console()
    OUTPUT_DIR.mkdir(exist_ok=True)
    Path(db_path).unlink(missing_ok=True)
    traces_path = OUTPUT_DIR / "model_swap_traces.jsonl"

    if offline:
        baselines_file = FIXTURES_DIR / "baselines.json"
        if not baselines_file.exists():
            console.print("[red]offline fixtures not found — run without --offline first[/red]")
            return
        with open(baselines_file) as f:
            baseline_outputs = json.load(f)
        with open(traces_path, "w") as out:
            for path in [FIXTURES_DIR / "traces_70b.jsonl", FIXTURES_DIR / "traces_8b.jsonl"]:
                with open(path) as f:
                    out.write(f.read())
    else:
        from spaniq.demos.generate_outputs import generate_outputs

        console.print("[bold]collecting 20 baseline outputs (70B)…[/bold]")
        baseline_outputs = generate_outputs(PROMPT, n=20, model=MODEL_70B, system_prompt=SYSTEM)
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        (FIXTURES_DIR / "baselines.json").write_text(json.dumps(baseline_outputs, indent=2))

        console.print("[bold]generating 20 normal traces (70B)…[/bold]")
        normal = generate_outputs(PROMPT, n=20, model=MODEL_70B, system_prompt=SYSTEM)
        with open(FIXTURES_DIR / "traces_70b.jsonl", "w") as f:
            for o in normal:
                f.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

        console.print("[bold]generating 20 swapped traces (8B)…[/bold]")
        swapped = generate_outputs(PROMPT, n=20, model=MODEL_8B, system_prompt=SYSTEM)
        with open(FIXTURES_DIR / "traces_8b.jsonl", "w") as f:
            for o in swapped:
                f.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

        with open(traces_path, "w") as out:
            for o in normal:
                out.write(json.dumps({"input": PROMPT, "output": o}) + "\n")
            for o in swapped:
                out.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

    store = BaselineStore(db_path)
    try:
        b = store.get_by_name("swap-demo")
        store.delete(b.id)
    except KeyError:
        pass
    store.create(name="swap-demo", prompt=PROMPT, outputs=baseline_outputs, model_name=MODEL_70B)

    console.print("\n[bold green]running monitor…[/bold green]")
    monitor = Monitor(
        baseline_name="swap-demo",
        collector=FileCollector(str(traces_path)),
        metrics=[
            ResponseDriftMetric(threshold=4.0, window_size=10),
            SemanticSimilarityMetric(),
            OutputStabilityMetric(threshold=0.15, window_size=10),
        ],
        db_path=db_path,
        alert_after=3,
        alerts_path=str(OUTPUT_DIR / "model_swap_alerts.jsonl"),
    )
    report = monitor.run()

    export_timeline_png(monitor.timeline_store, "OutputStabilityMetric",
                        output_path=str(OUTPUT_DIR / "model_swap_stability.png"), last_n=40)
    export_timeline_png(monitor.timeline_store, "ResponseDriftMetric",
                        output_path=str(OUTPUT_DIR / "model_swap_drift.png"), last_n=40)

    console.print("\n[bold]results:[/bold]")
    console.print(f"  total traces: {report.total_traces}")
    console.print(f"  alerts fired: {report.alerts_fired}")
    (OUTPUT_DIR / "model_swap_report.txt").write_text(
        f"model swap demo\ntraces: {report.total_traces}\n"
        f"alerts: {report.alerts_fired}\npass rates: {report.pass_rates}\n"
    )


if __name__ == "__main__":
    run()
