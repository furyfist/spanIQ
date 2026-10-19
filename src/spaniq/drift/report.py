"""Render a DiffResult for the terminal or as Markdown for a pull request comment."""

from __future__ import annotations

from spaniq.drift.detector import SignalDrift
from spaniq.drift.diff import DiffResult


def _verdict(result: DiffResult) -> str:
    if not result.drifted:
        return "no drift detected"
    units = ", ".join(result.drifted_units)
    blame = f"; most upstream drifted component: {result.root_cause}" if result.root_cause else ""
    return f"drift in {units}{blame}"


def _shift(s: SignalDrift) -> str:
    if s.effect_size != s.effect_size:
        return f"MMD {s.statistic:.4f}"
    return (
        f"{s.baseline_mean:.3g} -> {s.candidate_mean:.3g} (d={s.effect_size:+.2f}, PSI {s.psi:.2f})"
    )


def render_markdown(result: DiffResult, max_signals: int = 8) -> str:
    lines = [
        "## spanIQ diff",
        "",
        f"**{_verdict(result)}** ({result.n_baseline} baseline vs {result.n_candidate} candidate traces)",
        "",
    ]
    for u in result.units:
        drifted = u.report.drifted_signals
        mark = "drifted" if drifted else "stable"
        lines.append(f"### {u.unit}: {mark}")
        if drifted:
            lines += ["", "| signal | shift | p (FDR) |", "|---|---|---|"]
            for s in drifted[:max_signals]:
                lines.append(f"| {s.signal} | {_shift(s)} | {s.p_value:.2g} |")
        lines.append("")
    if result.review:
        lines += ["### Review first", ""]
        for r in result.review:
            output = r.output.replace("\n", " ")[:160]
            lines.append(f"- `{r.trace_id}` {'; '.join(r.reasons)}  \n  > {output}")
        lines.append("")
    return "\n".join(lines)


def print_diff(result: DiffResult, max_signals: int = 8) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    color = "red" if result.drifted else "green"
    console.print(
        f"[bold {color}]{_verdict(result)}[/bold {color}] "
        f"({result.n_baseline} baseline vs {result.n_candidate} candidate traces)"
    )
    for u in result.units:
        drifted = u.report.drifted_signals
        if not drifted:
            console.print(f"  [green]{u.unit}[/green]: stable")
            continue
        table = Table(title=f"{u.unit}: drifted", title_justify="left")
        table.add_column("signal")
        table.add_column("shift")
        table.add_column("p (FDR)", justify="right")
        for s in drifted[:max_signals]:
            table.add_row(s.signal, _shift(s), f"{s.p_value:.2g}")
        console.print(table)
    if result.review:
        console.print("\n[bold]review first[/bold]")
        for r in result.review:
            console.print(f"  {r.trace_id}  {'; '.join(r.reasons)}")
            console.print(f"    [dim]{r.output.replace(chr(10), ' ')[:140]}[/dim]")
