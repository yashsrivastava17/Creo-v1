from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.llm.router import Router
from jarvis.telemetry.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RouterModeController:
    router: Router

    def set_mode(self, smart: bool) -> dict[str, Any]:
        target = "gemini" if smart else "ollama"
        try:
            self.router.set_default(target)
        except ValueError as exc:
            LOGGER.warning("router.mode.unavailable", target=target, error=str(exc))
            target = self.router.current_provider()
        state = {"smart": target == "gemini", "provider": target}
        LOGGER.info("router.mode.set", state=state)
        return state

    def current(self) -> dict[str, Any]:
        provider = self.router.current_provider()
        return {"smart": provider == "gemini", "provider": provider}


_CONTROLLER: RouterModeController | None = None


def attach_router(router: Router) -> RouterModeController:
    global _CONTROLLER
    _CONTROLLER = RouterModeController(router)
    return _CONTROLLER


def set_mode(payload: dict[str, Any]) -> dict[str, Any]:
    if _CONTROLLER is None:
        raise RuntimeError("Router controller not initialised")
    return _CONTROLLER.set_mode(bool(payload.get("smart", False)))


def current_mode() -> dict[str, Any]:
    if _CONTROLLER is None:
        raise RuntimeError("Router controller not initialised")
    return _CONTROLLER.current()


__all__ = ["RouterModeController", "attach_router", "set_mode", "current_mode"]
