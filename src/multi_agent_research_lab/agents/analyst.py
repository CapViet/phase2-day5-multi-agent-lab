"""Analyst agent.

Responsibility: turn raw research notes into structured insight - key claims, points of
agreement/disagreement, and an honest note on weak or missing evidence.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a critical ANALYST agent. Given research notes, extract the key claims, compare "
    "viewpoints, and explicitly flag claims with weak or missing evidence. Be concise."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        super().__init__(llm=llm, settings=settings, temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.record_error("analyst: no research notes available")
            return state

        user_prompt = (
            f"Question: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Produce: (1) 3-5 key claims, (2) agreements/disagreements across sources, "
            "(3) evidence gaps or weak claims to caveat."
        )
        response = self._call_llm(state, _SYSTEM, user_prompt)
        state.analysis_notes = response.content
        self._record(state, AgentName.ANALYST, response.content, response)
        state.add_trace_event("analyst.done", {"chars": len(response.content)})
        return state
