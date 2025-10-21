from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.llm.plan_schema import Plan


@dataclass(slots=True)
class ChatResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RoutingPlan:
    provider: "ChatProvider"
    plan: Plan
    user_prompt: str
    context: dict[str, Any]
    tools: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "plan": self.plan.to_trace(),
            "trace": self.trace,
            "tools": self.tools,
        }

    def compose_prompt(self, plan_results: list[dict[str, Any]]) -> str:
        from json import dumps

        context_chunks: list[str] = []
        profile = self.context.get("profile") or []
        if profile:
            context_chunks.append("Profile facts:\n" + "\n".join(f"- {item}" for item in profile))
        ephemerals = self.context.get("ephemeral") or []
        if ephemerals:
            context_chunks.append("Recent notes:\n" + "\n".join(f"- {item}" for item in ephemerals))
        supplemental = self.context.get("supplemental")
        if supplemental:
            context_chunks.append("Supplemental context:\n" + dumps(supplemental, ensure_ascii=False, indent=2))

        plan_block = dumps(self.plan.to_trace(), ensure_ascii=False, indent=2)
        results_block = dumps(plan_results, ensure_ascii=False, indent=2) if plan_results else "[]"
        context_block = "\n\n".join(context_chunks) if context_chunks else "None"

        return (
            "You are Jarvis, a concise yet thorough assistant. "
            "Ground responses in the supplied context when possible.\n\n"
            f"<USER_PROMPT>\n{self.user_prompt}\n</USER_PROMPT>\n\n"
            f"<CONTEXT>\n{context_block}\n</CONTEXT>\n\n"
            f"<PLAN>\n{plan_block}\n</PLAN>\n\n"
            f"<PLAN_RESULTS>\n{results_block}\n</PLAN_RESULTS>\n\n"
            "Produce a final answer. Cite sources inline when context snippets are used. "
            "If information is missing, acknowledge limits instead of guessing."
        )

    def followup_prompt(self, tool_results: list[dict[str, Any]]) -> str:
        from json import dumps

        return (
            f"{self.compose_prompt(plan_results=[])}\n\n"
            f"Additional tool results:\n{dumps(tool_results, ensure_ascii=False, indent=2)}"
        )


class ChatProvider:
    name: str

    async def chat(self, prompt: str, tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        raise NotImplementedError
