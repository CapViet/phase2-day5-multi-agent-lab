"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _fmt(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)


def render_markdown_report(metrics: list[BenchmarkMetrics], title: str = "Benchmark Report") -> str:
    """Render benchmark metrics to a markdown table with a short takeaway."""

    lines = [
        f"# {title}",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = _fmt(item.estimated_cost_usd, ".4f")
        quality = _fmt(item.quality_score, ".1f")
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | "
            f"{coverage} | {failure} | {item.notes} |"
        )

    takeaway = _takeaway(metrics)
    if takeaway:
        lines += ["", "## Takeaway", "", takeaway]
    return "\n".join(lines) + "\n"


def _takeaway(metrics: list[BenchmarkMetrics]) -> str:
    """One-line comparison if we have both a baseline-ish and a multi-agent run."""

    if len(metrics) < 2:
        return ""
    fastest = min(metrics, key=lambda m: m.latency_seconds)
    best_cov = max(
        metrics, key=lambda m: (m.citation_coverage or 0.0, -m.latency_seconds)
    )
    return (
        f"Fastest run: **{fastest.run_name}** ({fastest.latency_seconds:.2f}s). "
        f"Best citation coverage: **{best_cov.run_name}** "
        f"({(best_cov.citation_coverage or 0.0):.0%}). "
        "Multi-agent typically trades higher latency/cost for better grounding and coverage; "
        "prefer it when the task decomposes cleanly and source attribution matters."
    )
