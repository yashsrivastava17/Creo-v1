from __future__ import annotations

from dataclasses import dataclass

from jarvis.llm.providers.gemini import GeminiProvider
from jarvis.llm.providers.ollama import OllamaProvider
from jarvis.llm.types import RoutingPlan
from jarvis.telemetry.logging import get_logger


@dataclass
class Router:
    ollama: OllamaProvider
    gemini: GeminiProvider | None

    def __post_init__(self) -> None:
        self._logger = get_logger(__name__)
        self._default_provider = "ollama"

    async def decide(self, prompt_text: str, context: dict) -> RoutingPlan:
        provider = self._select_provider()
        tools = context.get("tools_allowed", [])
        if "web.search" in tools and self.gemini:
            provider = self.gemini
        self._logger.info("llm.router.decision", provider=provider.name, prompt_len=len(prompt_text))
        return RoutingPlan(provider=provider, prompt=prompt_text, tools=[])

    def _select_provider(self) -> OllamaProvider | GeminiProvider:
        if self._default_provider == "gemini" and self.gemini is not None:
            return self.gemini
        return self.ollama

    def set_default(self, provider: str) -> str:
        provider = provider.lower()
        if provider == "gemini" and self.gemini is None:
            raise ValueError("Gemini provider not configured")
        if provider not in {"ollama", "gemini"}:
            raise ValueError("Unknown provider")
        self._default_provider = provider
        return self._default_provider

    def current_provider(self) -> str:
        return self._default_provider

    def available(self) -> list[str]:
        providers = ["ollama"]
        if self.gemini is not None:
            providers.append("gemini")
        return providers
