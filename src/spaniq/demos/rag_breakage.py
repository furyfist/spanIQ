"""Demo 3: RAG Context Breakage Detection

Baseline: 20 outputs generated WITH retrieval context (policy text in system prompt).
Normal traces: 20 more WITH context.
Broken traces: 20 WITHOUT context (simulating retrieval failure).

Expected: SemanticSimilarity drops (answers diverge from grounded baselines).
ResponseDrift detects hedging words ("I'm not sure", "generally", "typically").
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "rag_breakage"
OUTPUT_DIR = Path(__file__).parent / "output"

PROMPT = "What is the refund policy for orders placed in the last 30 days?"
CONTEXT = (
    "POLICY DOCUMENT: Customers may request a full refund within 30 days of purchase. "
    "Refunds are processed within 5-7 business days to the original payment method. "
    "Items must be in original condition. Contact support@store.com to initiate."
)
SYSTEM_WITH_CONTEXT = f"You are a customer support agent. Use the following policy to answer:\n\n{CONTEXT}"
SYSTEM_NO_CONTEXT = "You are a customer support agent. Answer the customer's question."
MODEL = "llama-3.3-70b-versatile"


def run(offline: bool = False, db_path: str = "spaniq_demo_rag.db") -> None:
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
    traces_path = OUTPUT_DIR / "rag_breakage_traces.jsonl"

    if offline:
        baselines_file = FIXTURES_DIR / "baselines.json"
        if not baselines_file.exists():
            console.print("[red]offline fixtures not found — run without --offline first[/red]")
            return
        with open(baselines_file) as f:
            baseline_outputs = json.load(f)
        with open(traces_path, "w") as out:
            for path in [FIXTURES_DIR / "traces_with_context.jsonl", FIXTURES_DIR / "traces_no_context.jsonl"]:
                with open(path) as f:
                    out.write(f.read())
    else:
        from spaniq.demos.generate_outputs import generate_outputs

        console.print("[bold]collecting 20 baseline outputs (with context)…[/bold]")
        baseline_outputs = generate_outputs(PROMPT, n=20, model=MODEL, system_prompt=SYSTEM_WITH_CONTEXT)
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        (FIXTURES_DIR / "baselines.json").write_text(json.dumps(baseline_outputs, indent=2))

        console.print("[bold]generating 20 normal traces (with context)…[/bold]")
        with_ctx = generate_outputs(PROMPT, n=20, model=MODEL, system_prompt=SYSTEM_WITH_CONTEXT)
        with open(FIXTURES_DIR / "traces_with_context.jsonl", "w") as f:
            for o in with_ctx:
                f.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

        console.print("[bold]generating 20 broken traces (no context)…[/bold]")
        no_ctx = generate_outputs(PROMPT, n=20, model=MODEL, system_prompt=SYSTEM_NO_CONTEXT)
        with open(FIXTURES_DIR / "traces_no_context.jsonl", "w") as f:
            for o in no_ctx:
                f.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

        with open(traces_path, "w") as out:
            for o in with_ctx:
                out.write(json.dumps({"input": PROMPT, "output": o}) + "\n")
            for o in no_ctx:
                out.write(json.dumps({"input": PROMPT, "output": o}) + "\n")

    store = BaselineStore(db_path)
    try:
        b = store.get_by_name("rag-demo")
        store.delete(b.id)
    except KeyError:
        pass
    store.create(name="rag-demo", prompt=PROMPT, outputs=baseline_outputs, model_name=MODEL)

    console.print("\n[bold green]running monitor…[/bold green]")
    monitor = Monitor(
        baseline_name="rag-demo",
        collector=FileCollector(str(traces_path)),
        metrics=[
            ResponseDriftMetric(threshold=0.1, window_size=10),
            SemanticSimilarityMetric(),
            OutputStabilityMetric(threshold=0.15, window_size=10),
        ],
        db_path=db_path,
        alert_after=3,
        alerts_path=str(OUTPUT_DIR / "rag_breakage_alerts.jsonl"),
    )
    report = monitor.run()

    export_timeline_png(monitor.timeline_store, "SemanticSimilarityMetric",
                        output_path=str(OUTPUT_DIR / "rag_breakage_semantic.png"), last_n=40)
    export_timeline_png(monitor.timeline_store, "ResponseDriftMetric",
                        output_path=str(OUTPUT_DIR / "rag_breakage_drift.png"), last_n=40)

    console.print(f"\n[bold]results:[/bold]")
    console.print(f"  total traces: {report.total_traces}")
    console.print(f"  alerts fired: {report.alerts_fired}")
    (OUTPUT_DIR / "rag_breakage_report.txt").write_text(
        f"rag breakage demo\ntraces: {report.total_traces}\nalerts: {report.alerts_fired}\npass rates: {report.pass_rates}\n"
    )


if __name__ == "__main__":
    run()
