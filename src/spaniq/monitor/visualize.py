from __future__ import annotations

from rich.console import Console

console = Console()


def export_timeline_png(
    timeline_store,
    metric_name: str,
    output_path: str = "timeline.png",
    last_n: int = 200,
    title: str | None = None,
) -> str:
    """Generate a matplotlib timeline chart as PNG. Returns output_path."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in CLI and tests
    import matplotlib.pyplot as plt

    rows = timeline_store.query(metric_name, last_n=last_n)
    if not rows:
        console.print(f"[yellow]no data for {metric_name}[/yellow]")
        return output_path

    scores = [r.score for r in rows]
    passed = [r.passed for r in rows]
    threshold = rows[0].threshold

    fig, ax = plt.subplots(figsize=(12, 4))
    x = list(range(len(scores)))

    ax.plot(x, scores, color="#2563eb", linewidth=1.5, zorder=3)
    ax.axhline(y=threshold, color="#dc2626", linestyle="--", linewidth=1, label=f"threshold ({threshold})")

    for i, p in enumerate(passed):
        if not p:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="#dc2626", zorder=1)

    ax.set_title(title or f"{metric_name} — last {len(scores)} traces")
    ax.set_xlabel("trace #")
    ax.set_ylabel("score")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_sparkline(
    timeline_store,
    metric_name: str,
    last_n: int = 50,
) -> None:
    """Print a unicode sparkline of recent scores to the terminal."""
    rows = timeline_store.query(metric_name, last_n=last_n)
    if not rows:
        console.print(f"[yellow]{metric_name}: no data[/yellow]")
        return

    scores = [r.score for r in rows]
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(scores), max(scores)
    rng = mx - mn or 1.0
    sparkline = "".join(blocks[min(int((s - mn) / rng * 7), 7)] for s in scores)

    threshold = rows[0].threshold
    passed_count = sum(1 for r in rows if r.passed)
    pass_rate = passed_count / len(rows) * 100

    console.print(
        f"[bold cyan]{metric_name}[/bold cyan]: {sparkline}  "
        f"[dim]pass {pass_rate:.0f}%  threshold {threshold}[/dim]"
    )
