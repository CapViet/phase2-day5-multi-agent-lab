"""Researcher agent.

Responsibility: gather sources for the query and distil them into concise, citation-aware
research notes. It owns the search client; downstream agents never touch search directly.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM = (
    "You are a meticulous RESEARCH agent. Given a question and a set of sources, write tight, "
    "factual notes. Attribute each claim to a source by its [n] index. Do not invent sources."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(llm=llm, settings=settings, temperature=0.2)
        self.search = search or SearchClient(settings=self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        try:
            sources = self.search.search(
                state.request.query, max_results=state.request.max_sources
            )
        except Exception as exc:  # surface as a recoverable agent error
            state.record_error(f"researcher: search failed: {exc}")
            raise AgentExecutionError(str(exc)) from exc

        if not sources:
            state.record_error("researcher: no sources found")

        state.sources = sources
        catalogue = "\n".join(
            f"[{i + 1}] {doc.title}: {doc.snippet}" for i, doc in enumerate(sources)
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{catalogue}\n\n"
            "Write 4-6 bullet research notes. Tag each with its [n] source index."
        )
        response = self._call_llm(state, _SYSTEM, user_prompt)
        state.research_notes = response.content
        self._record(state, AgentName.RESEARCHER, response.content, response)
        state.add_trace_event("researcher.done", {"n_sources": len(sources)})
        return state
