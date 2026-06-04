from rich.console import Console
from rich.table import Table

from spaniq.core.eval_result import CostReport, EvalResult

# GPT-4o pricing as of June 2026, conservative estimate
_GPT4O_INPUT_PER_1K = 0.0025
_GPT4O_OUTPUT_PER_1K = 0.01
_AVG_TOKENS_PER_JUDGE_CALL = 800
_CALLS_PER_METRIC = 3


def estimate_llm_judge_cost(n_test_cases: int, n_metrics: int) -> float:
    total_calls = n_test_cases * n_metrics * _CALLS_PER_METRIC
    total_tokens = total_calls * _AVG_TOKENS_PER_JUDGE_CALL
    input_cost = (total_tokens * 0.6) / 1000 * _GPT4O_INPUT_PER_1K
    output_cost = (total_tokens * 0.4) / 1000 * _GPT4O_OUTPUT_PER_1K
    return round(input_cost + output_cost, 4)


def build_cost_report(n_test_cases: int, n_metrics: int) -> CostReport:
    return CostReport(
        n_test_cases=n_test_cases,
        n_metrics=n_metrics,
        spaniq_cost=0.0,
        llm_judge_cost=estimate_llm_judge_cost(n_test_cases, n_metrics),
    )


def print_report(result: EvalResult) -> None:
    console = Console()
    report = result.cost_report

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("spaniq run complete", "")
    table.add_row(f"{report.n_test_cases} test cases × {report.n_metrics} metrics", "")
    table.add_row("spaniq cost:", "[green]$0.00[/green] (0 API calls)")
    table.add_row(
        "LLM-judge equiv:",
        f"[yellow]~${report.llm_judge_cost:.2f}[/yellow] "
        f"({report.n_test_cases * report.n_metrics * _CALLS_PER_METRIC:,} calls)",
    )
    table.add_row("duration:", f"{result.duration_seconds:.2f}s")
    table.add_row("passed:", f"[green]{result.total_passed}[/green]/{result.total_cases}")
    table.add_row("failed:", f"[red]{result.total_failed}[/red]/{result.total_cases}")

    console.print(table)
