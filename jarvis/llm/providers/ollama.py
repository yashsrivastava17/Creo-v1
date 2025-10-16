from __future__ import annotations

import httpx

from jarvis.llm.types import ChatProvider, ChatResponse
from jarvis.persona import SYSTEM_PROMPT
from jarvis.telemetry.logging import get_logger


class OllamaProvider(ChatProvider):
    def __init__(self, host: str, model: str = "llama3.2") -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(base_url=self._host, timeout=60.0)
        self._logger = get_logger(__name__)
        self.name = "ollama"

    async def chat(self, prompt: str, tools: list[dict] | None = None) -> ChatResponse:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
        }
        self._logger.info("ollama.chat", payload=payload)
        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return ChatResponse(text=data.get("response", ""), tool_calls=[], memory_writes=[])

    async def aclose(self) -> None:
        await self._client.aclose()
