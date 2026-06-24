"""Supervisor / router.

The supervisor is deliberately rule-based: routing is the one decision that must be cheap,
deterministic, and easy to debug. It inspects which fields of the shared state are still
missing and picks the next worker, enforcing a hard iteration ceiling as the loop guardrail.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState

# Routes the supervisor can emit.
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        decision = self._decide(state)
        state.next_action = decision
        state.record_route(decision)
        state.add_trace_event(
            "supervisor.route",
            {
                "next": decision,
                "iteration": state.iteration,
                "has_research": bool(state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_answer": bool(state.final_answer),
            },
        )
        return state

    def _decide(self, state: ResearchState) -> str:
        """Pick the next route based on missing outputs and guardrails.

        - Researcher first (no notes yet).
        - Analyst once research notes exist.
        - Writer once analysis exists.
        - Stop when an answer exists, the iteration ceiling is hit, or failures pile up.
        """

        # Guardrail: never loop forever.
        if state.iteration >= self.settings.max_iterations:
            return ROUTE_DONE
        # Guardrail: if every step keeps failing, stop instead of burning the budget.
        if len(state.errors) >= self.settings.max_iterations:
            return ROUTE_DONE

        if not state.research_notes:
            return ROUTE_RESEARCHER
        if not state.analysis_notes:
            return ROUTE_ANALYST
        if not state.final_answer:
            return ROUTE_WRITER
        return ROUTE_DONE
