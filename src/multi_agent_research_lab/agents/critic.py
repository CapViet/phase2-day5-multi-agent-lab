"""Optional critic agent.

Responsibility: a lightweight quality gate over the final answer. It computes citation
coverage from the answer text and records findings on the shared state. Cheap, deterministic
checks live here directly; an LLM review is layered on when a real provider is available.
"""

from __future__ import annotations

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a CRITIC agent. Check the answer for unsupported claims, hallucinated sources, "
    "and safety issues. Reply with a short list of concrete findings, or 'No issues found.'"
)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        super().__init__(llm=llm, settings=settings, temperature=0.0)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            state.record_error("critic: no final answer to review")
            return state

        coverage = self._citation_coverage(state)
        findings = self._call_llm(
            state,
            _SYSTEM,
            f"Question: {state.request.query}\n\nAnswer:\n{state.final_answer}\n\n"
            f"Available source indices: 1..{len(state.sources)}",
        )
        self._record(state, AgentName.CRITIC, findings.content, findings)
        state.add_trace_event(
            "critic.done",
            {
                "citation_coverage": coverage,
                "n_sources": len(state.sources),
            },
        )
        return state

    @staticmethod
    def _citation_coverage(state: ResearchState) -> float:
        """Fraction of available sources actually referenced in the final answer."""

        if not state.sources:
            return 0.0
        answer = state.final_answer or ""
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        valid = {i + 1 for i in range(len(state.sources))}
        used = cited & valid
        return round(len(used) / len(valid), 3)
