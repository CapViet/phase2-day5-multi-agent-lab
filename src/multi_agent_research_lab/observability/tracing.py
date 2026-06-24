"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or the simple JSON traces produced here.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from multi_agent_research_lab.core.config import Settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    Yields a mutable span dict so callers can attach attributes during execution. To send
    spans to LangSmith/Langfuse/OTel, emit from inside this context manager.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def export_trace(trace: list[dict[str, Any]], path: str | Path) -> Path:
    """Write the collected trace events to a JSON file for inspection/sharing."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return out


def _event_run_type(name: str) -> Literal["llm", "tool", "chain"]:
    """Map a flat trace-event name to a LangSmith run type."""

    if name.endswith(".llm"):
        return "llm"
    if name.endswith(".done") or ".route" in name or name.endswith(".error"):
        return "tool"
    return "chain"


def push_trace_to_langsmith(
    events: list[dict[str, Any]],
    *,
    query: str,
    answer: str,
    settings: Settings,
    run_name: str = "multi_agent_research",
) -> str | None:
    """Replay a collected flat trace into LangSmith as a run tree and return its URL.

    The live pipeline records provider-agnostic events on the shared state; here we lift
    those into a parent/child run tree so the whole run is viewable (and shareable) in the
    LangSmith UI. Returns ``None`` when no API key is configured so callers can no-op
    cleanly in offline/CI runs.
    """

    api_key = settings.langsmith_api_key
    if not api_key:
        logger.info("LangSmith API key not set; skipping trace push.")
        return None

    try:
        from langsmith import Client
        from langsmith.run_trees import RunTree
    except ImportError:
        logger.warning("langsmith not installed; skipping trace push. `pip install langsmith`.")
        return None

    client = Client(api_key=api_key)
    # Create the project up front so ingestion and URL resolution both target an existing
    # project (get_url -> read_project otherwise races the lazy project creation on first post).
    with contextlib.suppress(Exception):  # already-exists (409) and similar are non-fatal here
        client.create_project(project_name=settings.langsmith_project)

    started = datetime.now(UTC)
    root = RunTree(
        name=run_name,
        run_type="chain",
        inputs={"query": query},
        project_name=settings.langsmith_project,
        ls_client=client,
        start_time=started,
    )
    for i, event in enumerate(events):
        name = str(event.get("name", f"event_{i}"))
        payload = event.get("payload", {})
        # Synthetic, monotonically increasing timestamps preserve event order in the UI
        # (the flat trace does not carry real per-event wall-clock times).
        stamp = started + timedelta(milliseconds=i)
        child = root.create_child(
            name=name,
            run_type=_event_run_type(name),
            inputs={},
            start_time=stamp,
        )
        child.end(outputs=payload if isinstance(payload, dict) else {"value": payload})

    root.end(outputs={"answer": answer})
    root.post(exclude_child_runs=False)
    root.wait()
    try:
        url = root.get_url()
    except Exception as exc:  # noqa: BLE001 - URL resolution must not fail an otherwise-good run
        logger.warning("Trace posted but URL resolution failed: %s", exc)
        url = f"https://smith.langchain.com/o/me/projects/p/{settings.langsmith_project}"
    logger.info("LangSmith trace posted: %s", url)
    return url
