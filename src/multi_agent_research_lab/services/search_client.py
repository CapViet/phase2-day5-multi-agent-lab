"""Search client abstraction for ResearcherAgent.

Uses Tavily when ``TAVILY_API_KEY`` is configured, otherwise returns a deterministic,
offline mock corpus so the research pipeline stays runnable without credentials.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def online(self) -> bool:
        return bool(self.settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Falls back to a deterministic mock when no provider key is present.
        """

        if self.settings.tavily_api_key:
            try:
                return self._tavily_search(query, max_results)
            except Exception as exc:  # pragma: no cover - network failure path
                logger.warning("Tavily search failed (%s); using offline mock.", exc)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        from tavily import TavilyClient  # type: ignore[import-not-found]  # optional dep

        client = TavilyClient(api_key=self.settings.tavily_api_key)
        payload = client.search(query=query, max_results=max_results)
        docs: list[SourceDocument] = []
        for item in payload.get("results", [])[:max_results]:
            docs.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score")},
                )
            )
        logger.info("Tavily returned %d sources for query=%r", len(docs), query)
        return docs

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Deterministic placeholder corpus keyed off the query terms."""

        terms = [t for t in query.replace(",", " ").split() if len(t) > 3][:4]
        topic = " ".join(terms) or query
        templates = [
            (
                f"Overview of {topic}",
                f"A survey-style overview of {topic}, covering definitions, common "
                "architectures, and where current approaches still fall short.",
            ),
            (
                f"State of the art in {topic}",
                f"Recent results and benchmarks relevant to {topic}, including reported "
                "metrics and the datasets used to obtain them.",
            ),
            (
                f"Practical guide to {topic}",
                f"An engineering-oriented walkthrough of building with {topic}, with "
                "guardrails, failure modes, and operational trade-offs.",
            ),
            (
                f"Critiques and limitations of {topic}",
                f"A critical look at {topic}: reproducibility concerns, cost, latency, "
                "and cases where simpler methods perform comparably.",
            ),
            (
                f"Case studies using {topic}",
                f"Applied case studies that deployed {topic} in production and the "
                "lessons learned about reliability and evaluation.",
            ),
        ]
        docs = [
            SourceDocument(
                title=title,
                url=f"https://example.org/{i}-{'-'.join(terms) or 'doc'}",
                snippet=snippet,
                metadata={"mock": True, "rank": i + 1},
            )
            for i, (title, snippet) in enumerate(templates[:max_results])
        ]
        logger.info("Offline mock search returned %d sources for query=%r", len(docs), query)
        return docs
