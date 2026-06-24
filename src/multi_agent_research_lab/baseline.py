"""Single-agent baseline.

One agent does everything in a single LLM call: it is given the gathered sources and asked
to research, analyse, and write in one shot. This is the control we benchmark the multi-agent
workflow against (latency, cost, and quality).
"""

from __future__ import annotations

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM = (
    "You are a single research assistant. Given a question and sources, research, analyse, and "
    "write a clear, well-structured answer for the audience. Cite sources by [n] and end with a "
    "'Sources:' list. Do not invent sources."
)


class SingleAgentBaseline:
    """Minimal one-agent pipeline used as the benchmark control."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(settings=self.settings, temperature=0.3)
        self.search = search or SearchClient(settings=self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("baseline.search"):
            state.sources = self.search.search(
                state.request.query, max_results=state.request.max_sources
            )
        catalogue = "\n".join(
            f"[{i + 1}] {doc.title}: {doc.snippet}" for i, doc in enumerate(state.sources)
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{catalogue}\n\n"
            "Write the final answer now."
        )
        with trace_span("baseline.llm", {"model": self.llm.model}) as span:
            response = self.llm.complete(_SYSTEM, user_prompt)
            span["attributes"]["output_tokens"] = response.output_tokens

        answer = response.content
        if state.sources and "Sources:" not in answer:
            refs = "\n".join(
                f"[{i + 1}] {doc.title}" + (f" - {doc.url}" if doc.url else "")
                for i, doc in enumerate(state.sources)
            )
            answer = f"{answer}\n\nSources:\n{refs}"

        state.final_answer = answer
        state.record_route("baseline")
        state.add_agent_result(
            AgentResult(
                agent=AgentName.WRITER,
                content=answer,
                metadata={
                    "usage": {
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    }
                },
            )
        )
        state.add_trace_event("baseline.done", {"n_sources": len(state.sources)})
        return state
