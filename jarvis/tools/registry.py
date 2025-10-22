from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from jarvis.llm import router_modes
from jarvis.telemetry.logging import get_logger
from jarvis.tools import calendar as calendar_tools
from jarvis.tools import meeting as meeting_tools
from jarvis.tools import planner as planner_tools
from jarvis.tools import reminders as reminders_tools
from jarvis.tools import sensors as sensors_tools
from jarvis.tools import tasks as tasks_tools


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


class CalendarRangeArgs(BaseModel):
    time_min: datetime
    time_max: datetime


class CalendarSearchArgs(CalendarRangeArgs):
    query: str | None = Field(default=None, max_length=160)


class CalendarCreateArgs(BaseModel):
    title: str = Field(..., max_length=200)
    start: datetime
    end: datetime
    location: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=800)
    attendees: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CalendarUpdateArgs(CalendarCreateArgs):
    id: str


class TaskSyncArgs(BaseModel):
    list_names: list[str] | None = None


class TaskAddArgs(BaseModel):
    title: str = Field(..., max_length=200)
    est_min: int | None = Field(default=None, ge=0, le=480)
    list_name: str | None = Field(default=None, max_length=100)


class TaskCompleteArgs(BaseModel):
    id: str


class TaskListArgs(BaseModel):
    list_names: list[str] | None = None
    status: str | None = Field(default=None, max_length=32)


class EmptyArgs(BaseModel):
    """Placeholder for tools that do not accept input."""


class ReminderAddArgs(BaseModel):
    when: datetime = Field(..., description="UTC timestamp")
    title: str = Field(..., max_length=160)
    payload: dict[str, Any] = Field(default_factory=dict)
    channel: str = Field(default="local", max_length=32)

    def payload_dict(self) -> dict[str, Any]:
        return {
            "when_iso": self.when.isoformat(),
            "title": self.title,
            "payload": self.payload,
            "channel": self.channel,
        }


class ReminderCancelArgs(BaseModel):
    id: str


class PlannerPlanDayArgs(BaseModel):
    date: datetime | None = None
    mode: str | None = Field(default=None, max_length=32)

    def payload_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.date:
            payload["date"] = self.date.date().isoformat()
        if self.mode:
            payload["mode"] = self.mode
        return payload


class MeetingEventArgs(BaseModel):
    event_id: str = Field(..., min_length=3)


class RouterModeArgs(BaseModel):
    smart: bool = False


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


def register_productivity_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="calendar.freebusy",
            request_model=CalendarRangeArgs,
            handler=lambda args, _: calendar_tools.get_calendar_client().freebusy(args.time_min, args.time_max),
        )
    )
    registry.register(
        ToolSpec(
            name="calendar.search",
            request_model=CalendarSearchArgs,
            handler=lambda args, _: calendar_tools.get_calendar_client().search(
                args.time_min, args.time_max, query=args.query
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="calendar.create",
            request_model=CalendarCreateArgs,
            handler=lambda args, _: calendar_tools.get_calendar_client().create(
                args.model_dump(mode="json", exclude_none=True)
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="calendar.update",
            request_model=CalendarUpdateArgs,
            handler=lambda args, _: calendar_tools.get_calendar_client().update(
                args.model_dump(mode="json", exclude_none=True)
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="tasks.sync_from_apple",
            request_model=TaskSyncArgs,
            handler=lambda args, _: tasks_tools.sync_from_apple(args.list_names),
            timeout_s=12.0,
        )
    )
    registry.register(
        ToolSpec(
            name="tasks.add",
            request_model=TaskAddArgs,
            handler=lambda args, _: tasks_tools.add(args.title, est_min=args.est_min, list_name=args.list_name),
        )
    )
    registry.register(
        ToolSpec(
            name="tasks.complete",
            request_model=TaskCompleteArgs,
            handler=lambda args, _: tasks_tools.complete(args.id),
        )
    )
    registry.register(
        ToolSpec(
            name="tasks.list",
            request_model=TaskListArgs,
            handler=lambda args, _: tasks_tools.list_tasks(args.list_names, args.status),
        )
    )
    registry.register(
        ToolSpec(
            name="reminders.add",
            request_model=ReminderAddArgs,
            handler=lambda args, _: reminders_tools.add(args.payload_dict()),
        )
    )
    registry.register(
        ToolSpec(
            name="reminders.cancel",
            request_model=ReminderCancelArgs,
            handler=lambda args, _: reminders_tools.cancel({"id": args.id}),
        )
    )
    registry.register(
        ToolSpec(
            name="planner.plan_day",
            request_model=PlannerPlanDayArgs,
            handler=lambda args, _: planner_tools.plan_day(args.payload_dict()),
            timeout_s=10.0,
        )
    )
    registry.register(
        ToolSpec(
            name="meeting.brief",
            request_model=MeetingEventArgs,
            handler=lambda args, _: meeting_tools.brief(args.event_id),
        )
    )
    registry.register(
        ToolSpec(
            name="meeting.start_notes",
            request_model=MeetingEventArgs,
            handler=lambda args, _: meeting_tools.start_notes(args.event_id),
        )
    )
    registry.register(
        ToolSpec(
            name="meeting.postprocess",
            request_model=MeetingEventArgs,
            handler=lambda args, _: meeting_tools.postprocess(args.event_id),
            timeout_s=12.0,
        )
    )
    registry.register(
        ToolSpec(
            name="sensors.screen_ocr",
            request_model=EmptyArgs,
            handler=lambda _args, _: sensors_tools.screen_ocr(),
        )
    )
    registry.register(
        ToolSpec(
            name="sensors.activity_ping",
            request_model=EmptyArgs,
            handler=lambda _args, _: sensors_tools.activity_ping(),
        )
    )
    registry.register(
        ToolSpec(
            name="router.set_mode",
            request_model=RouterModeArgs,
            handler=lambda args, ctx: router_modes.set_mode(args.model_dump()),
        )
    )


__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "ToolContext",
    "ToolError",
    "RegistryToolExecutor",
    "register_productivity_tools",
    "CalendarRangeArgs",
    "CalendarSearchArgs",
    "CalendarCreateArgs",
    "CalendarUpdateArgs",
    "TaskSyncArgs",
    "TaskAddArgs",
    "TaskCompleteArgs",
    "TaskListArgs",
    "EmptyArgs",
    "ReminderAddArgs",
    "ReminderCancelArgs",
    "PlannerPlanDayArgs",
    "MeetingEventArgs",
    "RouterModeArgs",
]
