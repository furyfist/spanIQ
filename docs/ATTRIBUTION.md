# spanIQ V3 — Component-Level Failure Attribution

## Scope: population-level vs single-trace attribution

There are two distinct attribution problems. spanIQ V3 solves only one of them:

| | Single-trace attribution | Population-level attribution (spanIQ V3) |
|---|---|---|
| Question | "THIS one run failed. Which step caused it?" | "My pipeline degraded across recent traffic. Which component broke, and when did it start?" |
| Input | One failed trajectory | Score time series across hundreds of traces |
| Method | LLM reads transcript and reasons | Changepoint detection on per-component metric series |
| Cost | $0.10–1.00 per diagnosis | $0.00 |
| Deterministic | No | Yes |

spanIQ V3 does NOT compete with LLM-based single-trace methods. A single failed trace where an agent made one bad logical decision produces no distributional signal — that genuinely needs an LLM or a human. V3 covers systemic degradation: when one component's output distribution shifts and the failure cascades downstream.

## How it works

V2 already produces, per trace, per metric, a score written to TimelineStore. A pipeline with N components produces N parallel score series. When the system degrades:

1. One component's score series shifts first (the root cause)
2. Downstream components' series shift later (the cascade)
3. The time ordering of the shifts IS the attribution

V3 uses CUSUM (online) and PELT (offline) — mature classical statistics — to detect where each series breaks.

## CUSUM (online detection)

Two-sided CUSUM (Page 1954). Updated per trace at stride intervals to reduce autocorrelation from rolling-window scoring. Gives fast alarms; the alarm index is recorded as detection latency metadata.

Parameters: `k = 0.5 * baseline_sigma`, `h` calibrated empirically to ARL0 >= 500.

## PELT (offline localization)

Exact segmentation via ruptures with RBF cost model (Killick et al. 2012). Runs on demand over stored timeline to give precise break indices. The attribution report always quotes PELT localization.

## The autocorrelation trap

Rolling-window scores are autocorrelated (consecutive scores share window members). Naive CUSUM on these series inflates false alarms. V3 mitigates with stride subsampling (default: stride = window_size // 2) and empirical h calibration on real normal series.

## Cascade logic

Changepoints are clustered within a proximity window (default: 10 traces). Within a cluster, the earliest component is the root-cause candidate; later components are cascade. Confidence = 0.6 * (broken_metrics / total_metrics) + 0.4 * (lead_gap / cluster_window). The report says "broke first" — never "caused" — because temporal precedence is not the same as causal inference.

## Validation results

See `validation/results/summary.md` for detection delay, localization error, and attribution accuracy measured on synthetic ground-truth series.
