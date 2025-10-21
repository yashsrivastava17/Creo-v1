from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from jarvis.telemetry.logging import get_logger


class ToolError(Exception):
    """Raised when a tool cannot be executed."""


@dataclass(slots=True)
class ToolContext:
    memory: Any | None = None
    index: Any | None = None
    router: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolSpec:
    name: str
    request_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]] | dict[str, Any]]
    timeout_s: float = 8.0


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._logger = get_logger(__name__)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool '{spec.name}' already registered")
        self._specs[spec.name] = spec
        self._logger.info("tool.registry.registered", tool=spec.name)

    def available(self) -> list[str]:
        return sorted(self._specs.keys())

    async def run(self, name: str, payload: dict[str, Any], *, context: ToolContext | None = None) -> dict[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            raise ToolError(f"Unknown tool '{name}'")
        ctx = context or ToolContext()
        try:
            args = spec.request_model.model_validate(payload)
        except ValidationError as exc:
            raise ToolError(f"Invalid payload for tool '{name}': {exc}") from exc

        async def _invoke() -> dict[str, Any]:
            result = spec.handler(args, ctx)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, dict):
                raise ToolError(f"Tool '{name}' returned non-dict result")
            return result

        try:
            return await asyncio.wait_for(_invoke(), timeout=spec.timeout_s)
        except asyncio.TimeoutError as exc:
            raise ToolError(f"Tool '{name}' timed out after {spec.timeout_s}s") from exc


class RegistryToolExecutor:
    def __init__(self, registry: ToolRegistry, context_factory: Callable[[], ToolContext]) -> None:
        self._registry = registry
        self._context_factory = context_factory
        self._logger = get_logger(__name__)

    async def invoke(self, call: dict[str, Any]) -> dict[str, Any]:
        tool = call.get("tool") or call.get("name")
        if not tool:
            raise ToolError("Tool call missing 'tool' field")
        payload = call.get("args") or call.get("arguments") or {}
        context = self._context_factory()
        self._logger.info("tool.invoke", tool=tool, origin=call.get("origin"), payload=payload)
        return await self._registry.run(tool, payload, context=context)


__all__ = ["ToolRegistry", "ToolSpec", "ToolContext", "ToolError", "RegistryToolExecutor"]
