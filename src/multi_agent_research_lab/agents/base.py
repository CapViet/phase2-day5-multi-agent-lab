"""Base agent contract.

Every agent reads and writes the shared :class:`ResearchState`. The base class wires in
the service clients and provides a single traced LLM helper so each worker stays focused on
its own responsibility (and token/cost accounting happens in one place).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(settings=self.settings, temperature=temperature)

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""

    def _call_llm(
        self, state: ResearchState, system_prompt: str, user_prompt: str
    ) -> LLMResponse:
        """Call the LLM inside a trace span and roll usage into the shared state."""

        with trace_span(f"{self.name}.llm", {"model": self.llm.model}) as span:
            response = self.llm.complete(system_prompt, user_prompt)
            span["attributes"]["input_tokens"] = response.input_tokens
            span["attributes"]["output_tokens"] = response.output_tokens
        state.add_trace_event(
            f"{self.name}.llm",
            {
                "provider": response.provider,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return response

    def _record(
        self, state: ResearchState, agent: AgentName, content: str, response: LLMResponse | None
    ) -> None:
        """Append an AgentResult (with usage metadata) to the shared state."""

        usage = {}
        if response is not None:
            usage = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        state.add_agent_result(
            AgentResult(agent=agent, content=content, metadata={"usage": usage})
        )
