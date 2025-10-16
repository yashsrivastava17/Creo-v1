from __future__ import annotations

import os

import httpx

from jarvis.llm.types import ChatProvider, ChatResponse
from jarvis.persona import SYSTEM_PROMPT
from jarvis.telemetry.logging import get_logger


class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro") -> None:
        self._client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            timeout=60.0,
        )
        self._api_key = api_key
        self._model = model
        self._logger = get_logger(__name__)
        self.name = "gemini"

    async def chat(self, prompt: str, tools: list[dict] | None = None) -> ChatResponse:
        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7},
        }
        resp = await self._client.post(
            f"/models/{self._model}:generateContent",
            params={"key": self._api_key},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        text = ""
        if "candidates" in data and data["candidates"]:
            text = data["candidates"][0]["content"]["parts"][0].get("text", "")
        return ChatResponse(text=text, tool_calls=[], memory_writes=[])

    async def aclose(self) -> None:
        await self._client.aclose()
