"""Behavioural tests for the completed agents and workflow.

These run fully offline: with no OPENAI_API_KEY (and LLM_PROVIDER unset/mock) the LLM and
search clients use their deterministic fallbacks, so the whole pipeline is reproducible.
"""

import os

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE, ROUTE_RESEARCHER, ROUTE_WRITER
from multi_agent_research_lab.baseline import SingleAgentBaseline
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


@pytest.fixture(autouse=True)
def _force_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the deterministic offline backends regardless of the host environment."""

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


def _settings() -> Settings:
    return Settings()


def test_supervisor_routes_to_researcher_when_empty() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    SupervisorAgent(settings=_settings()).run(state)
    assert state.next_action == ROUTE_RESEARCHER
    assert state.route_history == [ROUTE_RESEARCHER]


def test_supervisor_routes_to_writer_then_done() -> None:
    supervisor = SupervisorAgent(settings=_settings())
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    supervisor.run(state)
    assert state.next_action == ROUTE_WRITER

    state.final_answer = "done"
    supervisor.run(state)
    assert state.next_action == ROUTE_DONE


def test_supervisor_stops_at_iteration_ceiling() -> None:
    settings = Settings(MAX_ITERATIONS=2)
    supervisor = SupervisorAgent(settings=settings)
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 2
    supervisor.run(state)
    assert state.next_action == ROUTE_DONE


def test_multi_agent_workflow_produces_answer_offline() -> None:
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG and summarize it"))
    result = MultiAgentWorkflow(settings=_settings()).run(state)
    assert result.final_answer
    assert result.sources  # researcher gathered mock sources
    assert result.research_notes and result.analysis_notes
    # Supervisor visited every worker and then stopped.
    assert "researcher" in result.route_history
    assert result.route_history[-1] == ROUTE_DONE


def test_baseline_produces_answer_offline() -> None:
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG and summarize it"))
    result = SingleAgentBaseline(settings=_settings()).run(state)
    assert result.final_answer
    assert "Sources:" in result.final_answer


def test_offline_provider_is_used_without_key() -> None:
    assert os.environ.get("OPENAI_API_KEY") is None
    from multi_agent_research_lab.services.llm_client import LLMClient

    client = LLMClient(settings=_settings())
    assert client.provider == "mock"
    assert client.online is False
