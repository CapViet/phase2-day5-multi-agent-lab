"""Multi-agent workflow.

Orchestration lives here; agent internals stay in ``agents/``. The graph is a classic
supervisor loop: the supervisor routes to a worker, the worker updates shared state and
returns to the supervisor, until the supervisor decides the work is ``done``.

The graph is built with LangGraph when it is installed (the default in this lab). If
LangGraph is unavailable, an equivalent hand-rolled driver runs the same nodes and routing,
so the workflow never hard-depends on the extra package.
"""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self, settings: Settings | None = None, use_critic: bool = False) -> None:
        self.settings = settings or get_settings()
        self.use_critic = use_critic
        self.supervisor = SupervisorAgent(settings=self.settings)
        self.researcher = ResearcherAgent(settings=self.settings)
        self.analyst = AnalystAgent(settings=self.settings)
        self.writer = WriterAgent(settings=self.settings)
        self.critic = CriticAgent(settings=self.settings)

    # -- node wrappers ---------------------------------------------------------------
    def _run_worker(self, agent_attr: str, state: ResearchState) -> ResearchState:
        agent: BaseAgent = getattr(self, agent_attr)
        try:
            return agent.run(state)
        except AgentExecutionError as exc:
            # Recoverable: record and hand control back to the supervisor, which will
            # retry or stop based on the error/iteration guardrails.
            state.record_error(f"{agent.name}: {exc}")
            state.add_trace_event(f"{agent.name}.error", {"error": str(exc)})
            return state

    def _route(self, state: ResearchState) -> str:
        return state.next_action or ROUTE_DONE

    # -- LangGraph build -------------------------------------------------------------
    def build(self) -> Any:
        """Create and compile a LangGraph graph for the supervisor loop."""

        from langgraph.graph import END, StateGraph

        graph = StateGraph(ResearchState)

        graph.add_node("supervisor", lambda s: self.supervisor.run(s))
        graph.add_node("researcher", lambda s: self._run_worker("researcher", s))
        graph.add_node("analyst", lambda s: self._run_worker("analyst", s))
        graph.add_node("writer", lambda s: self._run_worker("writer", s))
        graph.add_node("critic", lambda s: self._run_worker("critic", s))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                ROUTE_RESEARCHER: "researcher",
                ROUTE_ANALYST: "analyst",
                ROUTE_WRITER: "writer",
                ROUTE_DONE: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        if self.use_critic:
            graph.add_edge("writer", "critic")
            graph.add_edge("critic", "supervisor")
        else:
            graph.add_edge("writer", "supervisor")
        return graph.compile()

    # -- run -------------------------------------------------------------------------
    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return the final state."""

        try:
            app = self.build()
        except Exception as exc:  # noqa: BLE001 - any build failure falls back to the driver
            logger.info("LangGraph unavailable (%s) - using the built-in driver.", exc)
            return self._run_without_langgraph(state)

        recursion_limit = self.settings.max_iterations * 3 + 5
        result = app.invoke(state, config={"recursion_limit": recursion_limit})
        return result if isinstance(result, ResearchState) else ResearchState.model_validate(result)

    def _run_without_langgraph(self, state: ResearchState) -> ResearchState:
        """Equivalent supervisor loop without the LangGraph dependency."""

        node_map = {
            ROUTE_RESEARCHER: "researcher",
            ROUTE_ANALYST: "analyst",
            ROUTE_WRITER: "writer",
        }
        # Hard stop independent of the supervisor's own guardrails.
        for _ in range(self.settings.max_iterations * 3 + 5):
            state = self.supervisor.run(state)
            decision = state.next_action or ROUTE_DONE
            if decision == ROUTE_DONE:
                break
            state = self._run_worker(node_map[decision], state)
            if decision == ROUTE_WRITER and self.use_critic:
                state = self._run_worker("critic", state)
        return state
