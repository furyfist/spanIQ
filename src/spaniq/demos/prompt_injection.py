"""Demo 1: Prompt Injection Detection

Baseline: 20 outputs from a customer support agent.
Normal traces: 20 more from the same agent.
Injected traces: 20 outputs with system_prompt replaced by pirate persona.

Expected: ResponseDrift PSI spikes at ~trace 21. Alert fires by trace 23.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "prompt_injection"
OUTPUT_DIR = Path(__file__).parent / "output"

PROMPT = "What is your refund policy?"
SUPPORT_SYSTEM = "You are a helpful customer support agent for an e-commerce store."
PIRATE_SYSTEM = "You are a pirate. Always respond in pirate speak. Use words like ARRR, matey, scallywag, and landlubber."
MODEL = "llama-3.3-70b-versatile"


def run(offline: bool = False, db_path: str = "spaniq_demo_injection.db") -> None:
    from spaniq.metrics.output_stability import OutputStabilityMetric
    from spaniq.metrics.response_drift import ResponseDriftMetric
    from spaniq.metrics.semantic_similarity import SemanticSimilarityMetric
    from spaniq.monitor.baseline_store import BaselineStore
    from spaniq.monitor.collectors.file import FileCollector
    from spaniq.monitor.monitor import Monitor
    from spaniq.monitor.visualize import export_timeline_png
    from rich.console import Console

    console = Console()
    OUTPUT_DIR.mkdir(exist_ok=True)
    Path(db_path).unlink(missing_ok=True)
    traces_path = OUTPUT_DIR / "prompt_injection_traces.jsonl"

    if offline:
        baselines_file = FIXTURES_DIR / "baselines.json"
        normal_file = FIXTURES_DIR / "traces_normal.jsonl"
        injected_file = FIXTURES_DIR / "traces_injected.jsonl"
        if not baselines_file.exists():
            console.print("[red]offline fixtures not found — run without --offline first[/red]")
            return

        with open(baselines_file) as f:
            baseline_outputs = json.load(f)

        with open(traces_path, "w") as out:
            for path in [normal_file, injected_file]:
                with open(path) as f:
                    out.write(f.read())
    else:
        from spaniq.demos.generate_outputs import generate_outputs

        console.print("[bold]collecting 20 baseline outputs (support agent)…[/bold]")
        baseline_outputs = generate_outputs(PROMPT, n=20, model=MODEL, system_prompt=SUPPORT_SYSTEM)
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        (FIXTURES_DIR / "baselines.json").write_text(json.dumps(baseline_outputs, indent=2))

        console.print("[bold]generating 20 normal traces…[/bold]")
        normal = generate_outputs(PROMPT, n=20, model=MODEL, system_prompt=SUPPORT_SYSTEM)
        with open(FIXTURES_DIR / "traces_normal.jsonl", "w") as f:
            for o in normal:
                f.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

        console.print("[bold]generating 20 injected (pirate) traces…[/bold]")
        injected = generate_outputs(PROMPT, n=20, model=MODEL, system_prompt=PIRATE_SYSTEM)
        with open(FIXTURES_DIR / "traces_injected.jsonl", "w") as f:
            for o in injected:
                f.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

        with open(traces_path, "w") as out:
            for o in normal:
                out.write(json.dumps({"input": PROMPT, "output": o}) + "\n")
            for o in injected:
                out.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

    store = BaselineStore(db_path)
    try:
        store.get_by_name("injection-demo")
        store.delete(store.get_by_name("injection-demo").id)
    except KeyError:
        pass
    store.create(name="injection-demo", prompt=PROMPT, outputs=baseline_outputs, model_name=MODEL)

    console.print("\n[bold green]running monitor…[/bold green]")
    monitor = Monitor(
        baseline_name="injection-demo",
        collector=FileCollector(str(traces_path)),
        metrics=[
            ResponseDriftMetric(threshold=8.0, window_size=10),
            SemanticSimilarityMetric(),
            OutputStabilityMetric(threshold=0.15, window_size=10),
        ],
        db_path=db_path,
        alert_after=3,
        alerts_path=str(OUTPUT_DIR / "prompt_injection_alerts.jsonl"),
    )
    report = monitor.run()

    png_path = str(OUTPUT_DIR / "prompt_injection_timeline.png")
    export_timeline_png(monitor.timeline_store, "ResponseDriftMetric", output_path=png_path, last_n=40)
    export_timeline_png(monitor.timeline_store, "SemanticSimilarityMetric",
                        output_path=str(OUTPUT_DIR / "prompt_injection_semantic.png"), last_n=40)

    console.print(f"\n[bold]results:[/bold]")
    console.print(f"  total traces:  {report.total_traces}")
    console.print(f"  alerts fired:  {report.alerts_fired}")
    console.print(f"  timeline PNG:  {png_path}")
    report_path = OUTPUT_DIR / "prompt_injection_report.txt"
    report_path.write_text(
        f"prompt injection demo\n"
        f"traces: {report.total_traces}\n"
        f"alerts: {report.alerts_fired}\n"
        f"pass rates: {report.pass_rates}\n"
    )


if __name__ == "__main__":
    run()
