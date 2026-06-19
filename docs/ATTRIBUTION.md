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

## Known resolution limits

PELT's localization precision is bounded by `min_size` (default: 10 traces). When two components' real break points are closer together than this bound — or when the scored series is short enough that evidence for a gap is statistically weak — PELT will return the same or near-identical indices for both. This is a property of the method, not a bug.

When identical indices occur, the system resolves ranking using independent evidence: first, breadth of broken metrics (more metrics broken signals stronger root-cause impact); second, CUSUM alarm timing (an independent, non-PELT signal of which component's scores shifted first). The result is the best available ranking given the data, and the lead-gap reported in the verdict will be zero or near-zero — which is honest. Claiming false index precision in this regime would be worse than reporting the tie.

This parallels the V2 documented limitation on PSI vs cosine: each metric captures different aspects of drift, and neither is universally superior. Using them as independent evidence sources for tie-breaking is correct; treating either as a ground truth is not.

If finer resolution is required, increase the number of traces collected before attributing (larger `last_n`) or reduce `min_size` at the cost of more spurious breaks.

## Validation results

See `validation/results/summary.md` for detection delay, localization error, and attribution accuracy measured on synthetic ground-truth series.
