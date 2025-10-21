from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PlanStep(BaseModel):
    """Single action emitted by the planner."""

    use: str = Field(..., description="Qualified tool or action name, e.g. retrieve.topk or respond")
    args: dict[str, Any] = Field(default_factory=dict, description="JSON arguments payload for the action")

    @property
    def is_tool(self) -> bool:
        return "." in self.use and not self.use.startswith("respond")


class Plan(BaseModel):
    """Structured plan emitted by the small planner model."""

    type: Literal["plan"] = "plan"
    need_big_reasoning: bool = False
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    latency_budget_ms: int = Field(1200, ge=250, le=10000)
    steps: list[PlanStep] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def ensure_final_respond(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if steps and steps[-1].use != "respond":
            steps.append(PlanStep(use="respond"))
        elif not steps:
            steps = [PlanStep(use="respond")]
        return steps

    def estimated_tokens(self, query: str | None = None) -> int:
        """Crude token estimate for routing decisions."""
        base = len(query or "") // 4
        step_cost = sum(int(math.log(len(step.args) + 1, 2) * 20) for step in self.steps if step.args)
        return max(base + step_cost, 1)

    def to_trace(self) -> dict[str, Any]:
        return {
            "need_big_reasoning": self.need_big_reasoning,
            "confidence": self.confidence,
            "latency_budget_ms": self.latency_budget_ms,
            "steps": [step.model_dump() for step in self.steps],
        }


__all__ = ["Plan", "PlanStep"]
