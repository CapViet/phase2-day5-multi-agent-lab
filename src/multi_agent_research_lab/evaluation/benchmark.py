"""Benchmark single-agent vs multi-agent runs.

Captures latency, token cost, citation coverage, and failure rate for each run so the two
approaches can be compared on more than vibes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def citation_coverage(state: ResearchState) -> float:
    """Fraction of available sources referenced by the final answer."""

    if not state.sources:
        return 0.0
    answer = state.final_answer or ""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid = {i + 1 for i in range(len(state.sources))}
    return round(len(cited & valid) / len(valid), 3)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Run ``runner(query)`` and capture metrics.

    A runner that raises is recorded as a failed run (failure_rate = 1.0) rather than
    crashing the whole benchmark sweep.
    """

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - benchmark must survive a single bad run
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=round(latency, 3),
            estimated_cost_usd=0.0,
            citation_coverage=0.0,
            failure_rate=1.0,
            notes=f"FAILED: {exc}",
        )
        return None, metrics

    latency = perf_counter() - started
    failed = state.final_answer is None or bool(state.errors)
    coverage = citation_coverage(state)
    notes = (
        f"sources={len(state.sources)} coverage={coverage:.0%} "
        f"in_tok={state.total_input_tokens} out_tok={state.total_output_tokens}"
        + (f" errors={len(state.errors)}" if state.errors else "")
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(state.total_cost_usd, 6),
        citation_coverage=coverage,
        failure_rate=1.0 if failed else 0.0,
        notes=notes,
    )
    return state, metrics
