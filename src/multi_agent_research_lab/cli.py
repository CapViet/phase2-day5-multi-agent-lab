"""Command-line entrypoint for the lab."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.baseline import SingleAgentBaseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_trace
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_baseline(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return SingleAgentBaseline().run(state)


def _run_multi(query: str, use_critic: bool = False) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow(use_critic=use_critic).run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    state = _run_baseline(query)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic", help="Enable optional critic agent")] = False,
    trace_out: Annotated[
        str | None, typer.Option("--trace-out", help="Write JSON trace to this path")
    ] = None,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = _run_multi(query, use_critic=critic)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(
        f"[dim]route: {' -> '.join(state.route_history)} | "
        f"tokens in/out: {state.total_input_tokens}/{state.total_output_tokens} | "
        f"cost: ${state.total_cost_usd:.6f}[/dim]"
    )
    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="red"))
    if trace_out:
        path = export_trace(state.trace, trace_out)
        console.print(f"[green]Trace written to {path}[/green]")


@app.command()
def benchmark(
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Query (repeatable). Defaults to a built-in set."),
    ] = None,
    critic: Annotated[
        bool, typer.Option("--critic", help="Enable critic in multi-agent run")
    ] = False,
    out: Annotated[
        str, typer.Option("--out", help="Report path under reports/")
    ] = "benchmark_report.md",
) -> None:
    """Benchmark single-agent vs multi-agent across one or more queries."""

    _init()
    queries = query or [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    all_metrics = []
    table = Table(title="Benchmark")
    for col in ("Run", "Latency(s)", "Cost($)", "Citation", "Failure"):
        table.add_column(col)

    for q in queries:
        _, m_base = run_benchmark(f"baseline :: {q[:32]}", q, _run_baseline)
        _, m_multi = run_benchmark(
            f"multi :: {q[:32]}", q, lambda qq: _run_multi(qq, use_critic=critic)
        )
        for m in (m_base, m_multi):
            all_metrics.append(m)
            table.add_row(
                m.run_name,
                f"{m.latency_seconds:.2f}",
                f"{(m.estimated_cost_usd or 0):.4f}",
                "" if m.citation_coverage is None else f"{m.citation_coverage:.0%}",
                "" if m.failure_rate is None else f"{m.failure_rate:.0%}",
            )

    console.print(table)
    report = render_markdown_report(all_metrics, title="Single-Agent vs Multi-Agent Benchmark")
    path = LocalArtifactStore().write_text(out, report)
    console.print(f"[green]Report written to {path}[/green]")


if __name__ == "__main__":
    app()
