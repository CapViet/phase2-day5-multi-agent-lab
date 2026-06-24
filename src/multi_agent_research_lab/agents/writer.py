"""Writer agent.

Responsibility: synthesise research + analysis into the final answer for the requested
audience, with an explicit Sources list so claims remain traceable.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a clear WRITER agent. Synthesise the analysis into a well-structured answer for "
    "the target audience. Keep claims grounded in the provided notes and cite sources by [n]."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        super().__init__(llm=llm, settings=settings, temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        analysis = state.analysis_notes or state.research_notes or ""
        if not analysis:
            state.record_error("writer: nothing to write from")
            return state

        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{analysis}\n\n"
            "Write the final answer. End with a 'Sources:' list using the [n] indices."
        )
        response = self._call_llm(state, _SYSTEM, user_prompt)

        answer = response.content
        # Always append a machine-built Sources block so citation coverage is auditable,
        # even when the model omits one.
        if state.sources and "Sources:" not in answer:
            refs = "\n".join(
                f"[{i + 1}] {doc.title}" + (f" - {doc.url}" if doc.url else "")
                for i, doc in enumerate(state.sources)
            )
            answer = f"{answer}\n\nSources:\n{refs}"

        state.final_answer = answer
        self._record(state, AgentName.WRITER, answer, response)
        state.add_trace_event("writer.done", {"chars": len(answer)})
        return state
