from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spaniq.attribution.changepoint.pelt import detect_changepoints
from spaniq.monitor.timeline_store import TimelineStore


@dataclass
class ComponentBreak:
    component: str
    break_trace_index: int
    cusum_alarm_index: int | None
    broken_metrics: list[str]
    confidence: float


@dataclass
class AttributionResult:
    event_window: tuple[int, int]
    root_cause: ComponentBreak | None
    cascade: list[ComponentBreak]
    healthy: list[str]
    verdict: str


def attribute(
    timeline: TimelineStore,
    components: list[str],
    metrics: list[str],
    last_n: int = 500,
    cluster_window: int = 10,
    pelt_penalty: float | None = None,
    cusum_alarms: dict[str, dict[str, int]] | None = None,
    warmup: int = 0,
) -> AttributionResult:
    """Run PELT on each (component, metric) series, cluster changepoints,
    rank by earliest break. cusum_alarms is optional metadata from online detection."""

    breaks_with_metrics: dict[str, dict[int, list[str]]] = {c: {} for c in components}
    for component in components:
        for metric in metrics:
            series = timeline.query_series(component=component, metric_name=metric, last_n=last_n)
            if len(series) < 20:
                continue
            arr = np.array(series, dtype=float)
            cps = detect_changepoints(arr, penalty=pelt_penalty)
            for cp in cps:
                breaks_with_metrics[component].setdefault(cp, []).append(metric)

    component_breaks: list[ComponentBreak] = []
    for component in components:
        cp_map = breaks_with_metrics[component]
        if not cp_map:
            continue
        earliest = min(cp_map.keys())
        broken_metrics = cp_map[earliest]
        cusum_alarm = None
        if cusum_alarms and component in cusum_alarms:
            comp_alarms = cusum_alarms[component]
            if comp_alarms:
                cusum_alarm = min(comp_alarms.values())
        cb = ComponentBreak(
            component=component,
            break_trace_index=earliest + warmup,
            cusum_alarm_index=cusum_alarm,
            broken_metrics=broken_metrics,
            confidence=0.0,
        )
        component_breaks.append(cb)

    if not component_breaks:
        healthy = list(components)
        return AttributionResult(
            event_window=(0, last_n),
            root_cause=None,
            cascade=[],
            healthy=healthy,
            verdict="no degradation detected",
        )

    component_breaks.sort(key=lambda b: b.break_trace_index)

    # chain-cluster: group breaks where each is within cluster_window of the previous
    clusters: list[list[ComponentBreak]] = []
    current_cluster: list[ComponentBreak] = [component_breaks[0]]
    for cb in component_breaks[1:]:
        if cb.break_trace_index - current_cluster[-1].break_trace_index <= cluster_window:
            current_cluster.append(cb)
        else:
            clusters.append(current_cluster)
            current_cluster = [cb]
    clusters.append(current_cluster)

    # pick the largest cluster as the main event; ties go to the latest cluster
    event_cluster = max(clusters, key=lambda c: (len(c), c[0].break_trace_index))
    # tie-break: earlier PELT index first; on tie prefer more broken metrics (wider impact = cause);
    # final tie-break: earlier CUSUM alarm
    event_cluster.sort(
        key=lambda b: (
            b.break_trace_index,
            -len(b.broken_metrics),
            b.cusum_alarm_index if b.cusum_alarm_index is not None else 99999,
        )
    )

    n_metrics = len(metrics)
    for i, cb in enumerate(event_cluster):
        metric_score = len(cb.broken_metrics) / max(n_metrics, 1)
        if i == 0:
            lead_gap = (
                event_cluster[1].break_trace_index - cb.break_trace_index
                if len(event_cluster) > 1
                else cluster_window
            )
        else:
            lead_gap = cb.break_trace_index - event_cluster[i - 1].break_trace_index
        gap_score = min(1.0, lead_gap / max(cluster_window, 1))
        cb.confidence = 0.6 * metric_score + 0.4 * gap_score

    root_cause = event_cluster[0]
    cascade = event_cluster[1:]
    healthy = [c for c in components if c not in {b.component for b in event_cluster}]

    all_indices = [b.break_trace_index for b in event_cluster]
    event_window = (min(all_indices), last_n)

    if len(event_cluster) > 1:
        lead = event_cluster[1].break_trace_index - root_cause.break_trace_index
        verdict = (
            f"{root_cause.component} broke first by {lead} trace(s); "
            f"{', '.join(b.component for b in cascade)} drift is cascade"
        )
    elif len(event_cluster) == 1:
        verdict = f"{root_cause.component} broke; no cascade detected"
    else:
        verdict = "no degradation detected"

    return AttributionResult(
        event_window=event_window,
        root_cause=root_cause,
        cascade=cascade,
        healthy=healthy,
        verdict=verdict,
    )
