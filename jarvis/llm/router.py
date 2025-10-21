from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.llm.providers.gemini import GeminiProvider
from jarvis.llm.providers.ollama import OllamaProvider
from jarvis.llm.plan_schema import Plan, PlanStep
from jarvis.llm.types import RoutingPlan
from jarvis.orchestrator.policies import RouterPolicies
from jarvis.telemetry.logging import get_logger


@dataclass
class Router:
    ollama: OllamaProvider
    gemini: GeminiProvider | None
    policies: RouterPolicies | None = None

    def __post_init__(self) -> None:
        self._logger = get_logger(__name__)
        self._default_provider = "ollama"
        self._policies = self.policies or RouterPolicies()

    async def decide(self, prompt_text: str, context: dict) -> RoutingPlan:
        plan = self._draft_plan(prompt_text, context)
        estimated_tokens = plan.estimated_tokens(prompt_text)
        tools_allowed = context.get("tools_allowed", [])
        escalation_reasons: list[str] = []

        if plan.need_big_reasoning:
            escalation_reasons.append("planner_flag")
        if plan.confidence < self._policies.min_confidence:
            escalation_reasons.append("low_confidence")
        if estimated_tokens > self._policies.max_small_tokens:
            escalation_reasons.append("token_pressure")

        near_cost_cap = self._policies.cost_budget.near_cap()
        cost_cap_exceeded = self._policies.cost_budget.exceeded()
        provider = self._select_provider()
        if cost_cap_exceeded:
            escalation_reasons.append("cost_cap_exceeded")
        if escalation_reasons and self.gemini and not near_cost_cap and not cost_cap_exceeded:
            provider = self.gemini

        if "web.search" in tools_allowed and self.gemini and not near_cost_cap and not cost_cap_exceeded:
            provider = self.gemini
            escalation_reasons.append("tool_requirement")

        trace = {
            "estimated_tokens": estimated_tokens,
            "latency_budget_ms": plan.latency_budget_ms,
            "escalation_reasons": escalation_reasons,
            "near_cost_cap": near_cost_cap,
            "cost_cap_exceeded": cost_cap_exceeded,
            "default_provider": self._default_provider,
        }
        self._logger.info(
            "llm.router.decision",
            provider=provider.name,
            prompt_len=len(prompt_text),
            trace=trace,
        )
        return RoutingPlan(
            provider=provider,
            plan=plan,
            user_prompt=prompt_text,
            context=context,
            tools=self._derive_tools(plan),
            trace=trace,
        )

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

    def register_cost(self, provider_name: str) -> None:
        if provider_name == "gemini":
            self._policies.cost_budget.register(self._policies.gemini_cost_estimate_usd)

    def _draft_plan(self, prompt_text: str, context: dict[str, Any]) -> Plan:
        lower_prompt = prompt_text.lower()
        needs_retrieval = any(
            keyword in lower_prompt
            for keyword in ("remember", "recall", "note", "spec", "doc", "meeting", "last time", "what did we")
        )
        if context.get("profile") or context.get("ephemeral"):
            needs_retrieval = True

        steps: list[PlanStep] = []
        filters: dict[str, Any] = {}
        active_lens = context.get("active_lens") or context.get("lens")
        if active_lens:
            filters["lens"] = active_lens

        if needs_retrieval:
            steps.append(
                PlanStep(
                    use="retrieve.topk",
                    args={
                        "q": prompt_text,
                        "k": 5,
                        "filters": filters,
                    },
                )
            )

        style = "concise"
        if len(prompt_text) > 600 or any(word in lower_prompt for word in ("explain", "detail", "why", "deep")):
            style = "detailed"
        steps.append(PlanStep(use="respond", args={"style": style}))

        confidence = 0.88
        if len(prompt_text) > 1200:
            confidence -= 0.2
        if any(word in lower_prompt for word in ("why", "explain", "debug", "fix bug")):
            confidence -= 0.1
        confidence = max(0.1, min(confidence, 0.99))

        need_big_reasoning = any(
            keyword in lower_prompt for keyword in ("code", "implement", "database schema", "complex")
        ) or len(prompt_text) // 4 > int(self._policies.max_small_tokens * 0.8)

        plan = Plan(
            need_big_reasoning=need_big_reasoning,
            confidence=confidence,
            latency_budget_ms=self._policies.choose_latency_budget(need_big_reasoning),
            steps=steps,
        )
        return plan

    def _derive_tools(self, plan: Plan) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for step in plan.steps:
            if step.is_tool:
                tools.append({"name": step.use})
        return tools
