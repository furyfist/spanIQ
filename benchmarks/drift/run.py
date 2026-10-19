"""Run the drift benchmark and write results.csv and summary.md.

python -m benchmarks.drift.run --runs 5
python -m benchmarks.drift.run --detectors legacy --scenarios control,model_swap
python -m benchmarks.drift.run --live --save-traces benchmarks/results_drift/traces
python -m benchmarks.drift.run --traces benchmarks/results_drift/traces
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from dataclasses import asdict

from benchmarks.config import DRIFT_RESULTS_DIR
from benchmarks.drift.detectors import DETECTORS
from benchmarks.drift.harness import (
    ScenarioResult,
    judge_cost_per_1k,
    judge_sampling_delay,
    replay,
    run_scenario,
    summarize,
)
from benchmarks.drift.scenarios import SCENARIOS, load_items
from benchmarks.runners._cost import token_cost

# a judge call on one RAG answer is roughly 600 prompt and 10 completion tokens
_JUDGE_CALL_USD = token_cost(600, 10)
_SAMPLING_RATES = (0.01, 0.05, 0.2)
_FLAG_PROBS = (0.3, 0.6, 0.9)


def _write_streams(directory: pathlib.Path, scenario: str, seed: int, streams) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    baseline, stream, change_at = streams
    path = directory / f"{scenario}_seed{seed}.json"
    path.write_text(
        json.dumps({"baseline": baseline, "stream": stream, "change_at": change_at}),
        encoding="utf-8",
    )


def _read_streams(directory: pathlib.Path, scenario: str, seed: int):
    data = json.loads((directory / f"{scenario}_seed{seed}.json").read_text(encoding="utf-8"))
    return data["baseline"], data["stream"], data["change_at"]


def _fmt(value, pattern: str = "{:.2f}") -> str:
    return "n/a" if value is None else pattern.format(value)


def render_summary(rows: list[dict], source: str) -> str:
    lines = [
        "# Drift benchmark",
        "",
        f"Source: {source}. Detection rate and delay are over runs where the scenario was "
        "switched on; delay counts traces after the change. False alarms are per 1,000 "
        "traces seen before any change (all traces for control).",
        "",
        "| detector | scenario | runs | detected | median delay | false alarms / 1k | root correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["scenario"], r["detector"])):
        lines.append(
            f"| {r['detector']} | {r['scenario']} | {r['runs']} | "
            f"{_fmt(r['detection_rate'], '{:.0%}')} | {_fmt(r['median_delay'], '{:.0f}')} | "
            f"{r['false_alarms_per_1k']:.1f} | {_fmt(r['root_accuracy'], '{:.0%}')} |"
        )
    lines += [
        "",
        "## Sampled LLM judge, modeled",
        "",
        "Expected traces after a change before a judge sampling a fraction of traffic has "
        "flagged 3 outputs, for different chances that the judge flags a changed output. "
        f"Cost assumes ${_JUDGE_CALL_USD:.6f} per call. This table is arithmetic, not a run.",
        "",
        "| sampling rate | cost / 1k traces | "
        + " | ".join(f"delay if p_flag={p}" for p in _FLAG_PROBS)
        + " |",
        "|---|---|" + "---|" * len(_FLAG_PROBS),
    ]
    for rate in _SAMPLING_RATES:
        delays = " | ".join(f"{judge_sampling_delay(rate, p):.0f}" for p in _FLAG_PROBS)
        lines.append(f"| {rate:.0%} | ${judge_cost_per_1k(rate, _JUDGE_CALL_USD):.4f} | {delays} |")
    return "\n".join(lines) + "\n"


def main(
    detectors: list[str] | None = None,
    scenarios: list[str] | None = None,
    runs: int = 5,
    output_dir: str | pathlib.Path = DRIFT_RESULTS_DIR,
    live: bool = False,
    traces_dir: str | None = None,
    save_traces: str | None = None,
) -> list[dict]:
    detectors = detectors or list(DETECTORS)
    scenarios = scenarios or list(SCENARIOS)
    for name in detectors:
        if name not in DETECTORS:
            raise ValueError(f"unknown detector '{name}' (valid: {', '.join(DETECTORS)})")
    for name in scenarios:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario '{name}' (valid: {', '.join(SCENARIOS)})")

    items = load_items()
    results: list[ScenarioResult] = []
    for scenario_name in scenarios:
        scenario = SCENARIOS[scenario_name]
        for seed in range(runs):
            if traces_dir:
                streams = _read_streams(pathlib.Path(traces_dir), scenario_name, seed)
            elif live:
                from benchmarks.drift.live import build_live_stream

                streams = build_live_stream(scenario_name, items, seed)
            elif save_traces:
                from benchmarks.drift.scenarios import build_stream

                streams = build_stream(scenario, items, seed)
            else:
                streams = None
            if streams is not None and save_traces:
                _write_streams(pathlib.Path(save_traces), scenario_name, seed, streams)
            for det_name in detectors:
                if streams is None:
                    result = run_scenario(DETECTORS[det_name], scenario, items, seed)
                else:
                    result = replay(DETECTORS[det_name](), scenario, seed, *streams)
                results.append(result)
                print(
                    f"  [{det_name}] {scenario_name} seed={seed} "
                    f"alarm={result.first_alarm} delay={result.delay} "
                    f"false={result.false_alarms} root={result.found_root}"
                )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    rows = summarize(results)
    source = "real model calls" if (live or traces_dir) else "synthetic streams"
    (out / "summary.md").write_text(render_summary(rows, source), encoding="utf-8")
    print(f"wrote {out / 'results.csv'} and {out / 'summary.md'}")
    return rows


def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m benchmarks.drift.run")
    p.add_argument("--detectors", default=",".join(DETECTORS))
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", default=str(DRIFT_RESULTS_DIR))
    p.add_argument("--live", action="store_true", help="generate streams with real model calls")
    p.add_argument("--traces", default=None, help="replay streams saved by --save-traces")
    p.add_argument("--save-traces", default=None, dest="save_traces")
    a = p.parse_args()
    main(
        detectors=[d.strip() for d in a.detectors.split(",") if d.strip()],
        scenarios=[s.strip() for s in a.scenarios.split(",") if s.strip()],
        runs=a.runs,
        output_dir=a.output,
        live=a.live,
        traces_dir=a.traces,
        save_traces=a.save_traces,
    )


if __name__ == "__main__":
    _cli()
