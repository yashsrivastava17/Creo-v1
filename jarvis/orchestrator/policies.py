from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CostBudget:
    daily_cap_usd: float | None = None
    warning_ratio: float = 0.85
    _spent_today: float = 0.0
    _day_started: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def reset_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        if now.date() != self._day_started.date():
            self._spent_today = 0.0
            self._day_started = now

    def register(self, amount: float) -> None:
        self.reset_if_needed()
        self._spent_today += amount

    def near_cap(self) -> bool:
        if self.daily_cap_usd is None:
            return False
        self.reset_if_needed()
        return self._spent_today >= self.daily_cap_usd * self.warning_ratio

    def exceeded(self) -> bool:
        if self.daily_cap_usd is None:
            return False
        self.reset_if_needed()
        return self._spent_today >= self.daily_cap_usd


@dataclass
class RouterPolicies:
    min_confidence: float = 0.6
    max_small_tokens: int = 1500
    fast_path_latency_ms: int = 800
    slow_path_latency_ms: int = 2500
    gemini_cost_estimate_usd: float = 0.002  # rough per-call lower bound
    cost_budget: CostBudget = field(default_factory=CostBudget)

    def choose_latency_budget(self, need_big_reasoning: bool) -> int:
        return self.slow_path_latency_ms if need_big_reasoning else self.fast_path_latency_ms


@dataclass(slots=True)
class PlannerModeConfig:
    focus_minutes: int
    prep_buffer_min: int
    description: str = ""


@dataclass
class PlannerPolicies:
    default_mode: str = "balanced"
    microbreak_minutes: int = 5
    followup_minutes: int = 10
    workday_start_hour: int = 9
    workday_end_hour: int = 18
    modes: dict[str, PlannerModeConfig] = field(
        default_factory=lambda: {
            "speed": PlannerModeConfig(focus_minutes=30, prep_buffer_min=10, description="Quick triage"),
            "balanced": PlannerModeConfig(focus_minutes=45, prep_buffer_min=15, description="Default planning"),
            "deep": PlannerModeConfig(focus_minutes=60, prep_buffer_min=20, description="Deep work heavy"),
        }
    )

    def mode_settings(self, name: str) -> PlannerModeConfig:
        return self.modes.get(name, self.modes[self.default_mode])


__all__ = ["RouterPolicies", "CostBudget", "PlannerPolicies", "PlannerModeConfig"]
