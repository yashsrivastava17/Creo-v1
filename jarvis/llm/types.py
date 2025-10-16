from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RoutingPlan:
    provider: "ChatProvider"
    prompt: str
    tools: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider.name, "tools": self.tools}

    def followup_prompt(self, tool_results: list[dict[str, Any]]) -> str:
        return f"{self.prompt}\nTool results:\n{tool_results}"


class ChatProvider:
    name: str

    async def chat(self, prompt: str, tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        raise NotImplementedError

