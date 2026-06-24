"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.

The client is provider-pluggable and switchable with a single setting (``LLM_PROVIDER``):

* ``openai`` - the hosted OpenAI Chat Completions API.
* ``ollama`` - a local Ollama server via its OpenAI-compatible endpoint (free, offline-capable).
* ``mock``   - a deterministic, network-free synthesiser so the lab and tests run with no setup.
* ``auto``   - OpenAI if a key is configured, else a reachable local Ollama, else ``mock``.

Switching the *model* is just ``OPENAI_MODEL`` / ``OLLAMA_MODEL`` (or per-call ``model=``).
Retry, timeout, token accounting, and cost estimation all live here rather than in agents.
"""

from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Approximate public pricing per 1M tokens: (input, output) in USD. Local models are free.
_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}
_DEFAULT_PRICING = (0.15, 0.60)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    provider: str = "mock"
    model: str = ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used for offline cost accounting."""

    return max(1, len(text) // 4)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _PRICING_PER_M.get(model, _DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def _ollama_reachable(base_url: str, timeout: float = 0.4) -> bool:
    """Cheap TCP probe so 'auto' can detect a local Ollama without importing the SDK."""

    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class LLMClient:
    """Provider-agnostic LLM client.

    Parameters
    ----------
    settings:
        Injected settings. Falls back to the cached global settings.
    temperature:
        Per-agent override so e.g. the writer can run hotter than the supervisor.
    model:
        Optional explicit model override. When omitted the model is taken from the
        resolved provider's setting, so switching models is a one-line config change.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.temperature = temperature
        self.provider = self._resolve_provider()
        self.model = model or self._default_model()
        self._client: Any = self._build_client()
        logger.info("LLMClient ready: provider=%s model=%s", self.provider, self.model)

    def _resolve_provider(self) -> str:
        choice = (self.settings.llm_provider or "auto").lower()
        if choice in {"openai", "ollama", "mock"}:
            return choice
        # auto
        if self.settings.openai_api_key:
            return "openai"
        if _ollama_reachable(self.settings.ollama_base_url):
            return "ollama"
        return "mock"

    def _default_model(self) -> str:
        if self.provider == "ollama":
            return self.settings.ollama_model
        return self.settings.openai_model

    def _build_client(self) -> Any:
        if self.provider == "mock":
            logger.info("LLMClient running in offline mock mode.")
            return None
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed - falling back to offline mock mode.")
            self.provider = "mock"
            return None
        if self.provider == "ollama":
            # Ollama exposes an OpenAI-compatible API; the api_key is ignored but required.
            return OpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
                timeout=float(self.settings.timeout_seconds),
            )
        return OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=float(self.settings.timeout_seconds),
        )

    @property
    def online(self) -> bool:
        """True when a real provider connection is available (OpenAI or Ollama)."""

        return self._client is not None

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Routes to the configured provider, otherwise to the deterministic offline
        synthesiser. Retry/timeout/token-logging are handled here so agents stay thin.
        """

        if self._client is None:
            return self._offline_complete(system_prompt, user_prompt)
        return self._online_complete(system_prompt, user_prompt)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def _online_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        # Local models are free; hosted models use the pricing table.
        cost: float | None = 0.0 if self.provider == "ollama" else None
        if self.provider != "ollama" and input_tokens is not None and output_tokens is not None:
            cost = _estimate_cost(self.model, input_tokens, output_tokens)
        logger.info(
            "LLM call provider=%s model=%s in_tokens=%s out_tokens=%s cost_usd=%s",
            self.provider,
            self.model,
            input_tokens,
            output_tokens,
            None if cost is None else f"{cost:.6f}",
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            provider=self.provider,
            model=self.model,
        )

    def _offline_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic, network-free synthesis.

        It condenses the supplied context into a readable block so downstream agents
        receive meaningful, reproducible text. This keeps the lab runnable for everyone
        while leaving the real-provider path one config flag away.
        """

        content = _offline_synthesise(system_prompt, user_prompt)
        input_tokens = _estimate_tokens(system_prompt + user_prompt)
        output_tokens = _estimate_tokens(content)
        cost = _estimate_cost(self.model, input_tokens, output_tokens)
        logger.debug(
            "Offline LLM synthesis in_tokens=%s out_tokens=%s", input_tokens, output_tokens
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            provider="mock",
            model=self.model,
        )


def _offline_synthesise(system_prompt: str, user_prompt: str) -> str:
    """Produce deterministic prose from the prompt context.

    Each agent's system prompt carries an unambiguous role token (e.g. ``RESEARCH agent``).
    We branch on those tokens so each agent gets shaped output without any external model.
    """

    sp = system_prompt
    bullets = _extract_salient_lines(user_prompt, limit=6)
    joined = "\n".join(f"- {b}" for b in bullets) if bullets else "- (no context provided)"

    if "RESEARCH agent" in sp:
        return (
            "Research notes (offline synthesis):\n"
            f"{joined}\n"
            "Open questions: verify recency of refs and quantify any benchmark claims."
        )
    if "ANALYST agent" in sp:
        return (
            "Analysis (offline synthesis):\n"
            "Key claims and how well they are supported:\n"
            f"{joined}\n"
            "Confidence: medium. Weakest link: claims without a cited primary reference."
        )
    if "CRITIC agent" in sp:
        return "No blocking issues found (offline critic). Verify [n] citations against sources."
    if "WRITER agent" in sp or "research assistant" in sp:
        headline = bullets[0] if bullets else "See findings below."
        return (
            "Summary:\n"
            f"{headline}\n\n"
            "Key findings:\n"
            f"{joined}\n\n"
            "Citations map to the gathered references by index."
        )
    # Generic fallback.
    return "Response (offline synthesis):\n" + joined


def _extract_salient_lines(text: str, limit: int) -> list[str]:
    """Pick the most informative non-empty fragments from a prompt."""

    candidates: list[str] = []
    for raw in text.splitlines():
        line = raw.strip(" -*\t")
        if not line:
            continue
        # Skip pure section headers we add ourselves.
        if line.endswith(":") and len(line) < 40:
            continue
        candidates.append(line)
    if not candidates:
        # Fall back to sentence splitting on a single-line prompt.
        candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return candidates[:limit]
