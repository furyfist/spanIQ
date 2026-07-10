from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spaniq.attribution.attributor import AttributionResult, ComponentBreak
from spaniq.monitor.timeline_store import TimelineStore

console = Console()


def print_attribution(result: AttributionResult) -> None:
    """Print the attribution verdict to the terminal using rich."""
    if result.root_cause is None:
        console.print(Panel("no degradation detected", title="spanIQ attribution", style="green"))
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("component", style="bold")
    table.add_column("break")
    table.add_column("metrics")
    table.add_column("label")

    if result.root_cause:
        rc = result.root_cause
        label = Text("ROOT CAUSE", style="bold red")
        table.add_row(
            rc.component,
            f"trace #{rc.break_trace_index}",
            ", ".join(rc.broken_metrics),
            label,
        )

    for cb in result.cascade:
        lag = cb.break_trace_index - (
            result.root_cause.break_trace_index if result.root_cause else 0
        )
        label = Text(f"cascade, +{lag}", style="yellow")
        table.add_row(
            cb.component,
            f"trace #{cb.break_trace_index}",
            ", ".join(cb.broken_metrics),
            label,
        )

    for comp in result.healthy:
        table.add_row(comp, "no break", "", Text("healthy", style="green"))

    event_start, event_end = result.event_window
    panel_title = f"traces {event_start}–{event_end}"
    console.print(Panel(table, title=panel_title, border_style="red"))
    console.print(f"[bold]verdict:[/bold] {result.verdict}")
    console.print("[dim]diagnosis cost: $0.00[/dim]")


def attribution_to_dict(result: AttributionResult) -> dict:
    def cb_dict(cb: ComponentBreak) -> dict:
        return {
            "component": cb.component,
            "break_trace_index": cb.break_trace_index,
            "cusum_alarm_index": cb.cusum_alarm_index,
            "broken_metrics": cb.broken_metrics,
            "confidence": round(cb.confidence, 4),
        }

    return {
        "event_window": list(result.event_window),
        "root_cause": cb_dict(result.root_cause) if result.root_cause else None,
        "cascade": [cb_dict(cb) for cb in result.cascade],
        "healthy": result.healthy,
        "verdict": result.verdict,
    }


def save_attribution_json(result: AttributionResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(attribution_to_dict(result), f, indent=2)


def save_attribution_png(
    result: AttributionResult,
    timeline: TimelineStore,
    components: list[str],
    primary_metric: str,
    path: str,
    last_n: int = 500,
) -> None:
    import matplotlib.pyplot as plt

    n = len(components)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    all_breaks: dict[str, int | None] = {}
    if result.root_cause:
        all_breaks[result.root_cause.component] = result.root_cause.break_trace_index
    for cb in result.cascade:
        all_breaks[cb.component] = cb.break_trace_index

    for ax, component in zip(axes, components, strict=True):
        series = timeline.query_series(
            component=component, metric_name=primary_metric, last_n=last_n
        )
        xs = list(range(len(series)))
        color = "steelblue"
        if component == (result.root_cause.component if result.root_cause else None):
            color = "crimson"
        ax.plot(xs, series, color=color, linewidth=1.2, label=component)
        ax.set_ylabel(primary_metric[:20], fontsize=8)
        ax.set_title(component, fontsize=9, loc="left")

        if component in all_breaks and all_breaks[component] is not None:
            bp = all_breaks[component]
            lc = (
                "red"
                if component == (result.root_cause.component if result.root_cause else None)
                else "orange"
            )
            ax.axvline(x=bp, color=lc, linestyle="--", linewidth=1.5, label=f"break @{bp}")
            ax.legend(fontsize=7)

    axes[-1].set_xlabel("trace index")
    fig.suptitle(f"spanIQ attribution — {result.verdict[:80]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
