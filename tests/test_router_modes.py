from __future__ import annotations

from dataclasses import dataclass

from jarvis.llm import router_modes
from jarvis.llm.router import Router
from jarvis.orchestrator.policies import RouterPolicies


@dataclass
class StubProvider:
    name: str


def test_router_mode_switches() -> None:
    router = Router(ollama=StubProvider("ollama"), gemini=StubProvider("gemini"), policies=RouterPolicies())
    router_modes.attach_router(router)

    smart_state = router_modes.set_mode({"smart": True})
    assert smart_state["provider"] == "gemini"

    lite_state = router_modes.set_mode({"smart": False})
    assert lite_state["provider"] == "ollama"


def test_router_mode_handles_missing_gemini() -> None:
    router = Router(ollama=StubProvider("ollama"), gemini=None, policies=RouterPolicies())
    router_modes.attach_router(router)
    state = router_modes.set_mode({"smart": True})
    assert state["provider"] == "ollama"
