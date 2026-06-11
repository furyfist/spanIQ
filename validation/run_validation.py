"""Validation sweep: detection delay, localization error, false alarm rate, attribution accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import json
from dataclasses import dataclass, asdict

import numpy as np

from spaniq.attribution.changepoint.calibrate import calibrate_h
from spaniq.attribution.changepoint.cusum import run_cusum
from spaniq.attribution.changepoint.pelt import detect_changepoints, bic_penalty
sys.path.insert(0, str(Path(__file__).parent))
from synthetic_series import make_break_series, make_cascade_scenario, make_normal_series

RESULTS_DIR = Path(__file__).parent / "results"
N_SERIES = 200
N_NORMAL = 1000
SHIFT_SIGMAS = [0.5, 1.0, 2.0, 4.0]
PENALTIES = [1.0, 3.0, 5.0]
LEAD_GAPS = list(range(2, 16))


@dataclass
class DelayRow:
    shift_sigma: float
    penalty: float
    median_delay_cusum: float
    median_localization_error_pelt: float
    false_alarm_rate_per_1k: float


@dataclass
class AttributionRow:
    lead_gap: int
    accuracy: float
    n_scenarios: int


def _cusum_delay(series: np.ndarray, true_break: int, h: float, k_factor: float = 0.5) -> int | None:
    mu0 = float(np.mean(series[:true_break]))
    sigma0 = float(np.std(series[:true_break])) or 0.04
    k = k_factor * sigma0
    state = run_cusum(list(series), mu0=mu0, k=k, h=h)
    if state.alarm_index is None:
        return None
    return max(0, state.alarm_index - true_break)


def _pelt_localization(series: np.ndarray, true_break: int, penalty: float) -> int | None:
    cps = detect_changepoints(series, penalty=penalty)
    if not cps:
        return None
    closest = min(cps, key=lambda cp: abs(cp - true_break))
    return abs(closest - true_break)


def _false_alarm_rate(h: float, n_traces: int = 1000, k_factor: float = 0.5, n_series: int = 50) -> float:
    alarms = 0
    rng = np.random.default_rng(777)
    for seed in range(n_series):
        series = make_normal_series(n=n_traces, seed=seed)
        mu0 = float(np.mean(series))
        sigma0 = float(np.std(series)) or 0.04
        k = k_factor * sigma0
        state = run_cusum(list(series), mu0=mu0, k=k, h=h)
        if state.alarm_index is not None:
            alarms += 1
    return (alarms / n_series) * 1000 / n_traces


def run_sweep() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    normal_series = [make_normal_series(n=N_NORMAL, seed=i) for i in range(10)]
    mu0 = float(np.mean(np.concatenate(normal_series)))
    sigma0 = float(np.std(np.concatenate(normal_series))) or 0.04
    k = 0.5 * sigma0
    h = calibrate_h(normal_series, k=k, mu0=mu0, target_arl0=500)
    print(f"Calibrated h={h:.2f} (ARL0>=500)")

    delay_rows: list[DelayRow] = []
    for shift_sigma in SHIFT_SIGMAS:
        for penalty in PENALTIES:
            delays_cusum, errors_pelt = [], []
            for seed in range(N_SERIES):
                series, true_break = make_break_series(n=300, shift_sigma=shift_sigma, seed=seed)
                d = _cusum_delay(series, true_break, h=h)
                if d is not None:
                    delays_cusum.append(d)
                e = _pelt_localization(series, true_break, penalty=penalty)
                if e is not None:
                    errors_pelt.append(e)
            fa_rate = _false_alarm_rate(h=h)
            row = DelayRow(
                shift_sigma=shift_sigma,
                penalty=penalty,
                median_delay_cusum=float(np.median(delays_cusum)) if delays_cusum else float("nan"),
                median_localization_error_pelt=float(np.median(errors_pelt)) if errors_pelt else float("nan"),
                false_alarm_rate_per_1k=round(fa_rate, 2),
            )
            delay_rows.append(row)
            print(f"  shift={shift_sigma}s pen={penalty}: delay={row.median_delay_cusum:.0f} loc_err={row.median_localization_error_pelt:.0f} FA={row.false_alarm_rate_per_1k}")

    with open(RESULTS_DIR / "delay_localization.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(delay_rows[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in delay_rows])

    attr_rows: list[AttributionRow] = []
    for lead_gap in LEAD_GAPS:
        correct = 0
        total = 100
        for seed in range(total):
            root_s, follow_s, root_bp, follow_bp = make_cascade_scenario(
                n=300, root_break=100, lead_gap=lead_gap, shift_sigma=2.0, seed=seed
            )
            root_cps = detect_changepoints(root_s, penalty=3.0)
            follow_cps = detect_changepoints(follow_s, penalty=3.0)
            root_detected = min(root_cps, key=lambda cp: abs(cp - root_bp)) if root_cps else None
            follow_detected = min(follow_cps, key=lambda cp: abs(cp - follow_bp)) if follow_cps else None
            if root_detected is not None and follow_detected is not None:
                if root_detected <= follow_detected:
                    correct += 1
            elif root_detected is not None:
                correct += 1
        acc = correct / total
        attr_rows.append(AttributionRow(lead_gap=lead_gap, accuracy=acc, n_scenarios=total))
        print(f"  lead_gap={lead_gap}: attribution accuracy={acc*100:.0f}%")

    with open(RESULTS_DIR / "attribution_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(attr_rows[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in attr_rows])

    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({"calibrated_h": h, "k": k, "target_arl0": 500}, f, indent=2)

    print(f"\nResults written to {RESULTS_DIR}/")
    _write_summary(delay_rows, attr_rows, h)


def _write_summary(delay_rows, attr_rows, h: float) -> None:
    lines = [
        "# Validation Results",
        "",
        f"Calibrated CUSUM h = {h:.2f} (empirical, target ARL0 >= 500)",
        "",
        "## Detection Delay (CUSUM) + Localization Error (PELT, pen=3.0)",
        "",
        "| shift | delay (median traces) | localization error (median) | false alarms /1k |",
        "|---|---|---|---|",
    ]
    for r in [row for row in delay_rows if row.penalty == 3.0]:
        lines.append(f"| {r.shift_sigma}s | {r.median_delay_cusum:.0f} | ±{r.median_localization_error_pelt:.0f} | {r.false_alarm_rate_per_1k} |")

    lines += [
        "",
        "## Attribution Accuracy (root component ranked first)",
        "",
        "| lead gap | accuracy |",
        "|---|---|",
    ]
    for r in attr_rows:
        lines.append(f"| {r.lead_gap} traces | {r.accuracy*100:.0f}% |")

    ge5 = [r.accuracy for r in attr_rows if r.lead_gap >= 5]
    lt5 = [r.accuracy for r in attr_rows if r.lead_gap < 5]
    lines += [
        "",
        f"Attribution accuracy (lead gap >= 5): {np.mean(ge5)*100:.0f}%",
        f"Attribution accuracy (lead gap < 5): {np.mean(lt5)*100:.0f}% (documented limitation)",
    ]

    with open(RESULTS_DIR / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"summary written to {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    run_sweep()
