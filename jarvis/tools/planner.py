from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from jarvis.orchestrator.policies import PlannerPolicies
from jarvis.telemetry.logging import get_logger
from jarvis.tools import calendar as calendar_tools
from jarvis.tools import reminders as reminder_tools
from jarvis.tools import tasks as task_tools

LOGGER = get_logger(__name__)
@dataclass(slots=True)
class PlanBlock:
    type: str
    start: datetime
    end: datetime
    label: str
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "label": self.label,
            "meta": self.extra,
        }
class DayPlanner:
    def __init__(self, policies: PlannerPolicies | None = None) -> None:
        self._policies = policies or PlannerPolicies()
        self._calendar = calendar_tools.get_calendar_client()
        self._tasks = task_tools.get_store()

    def plan_day(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_date = self._parse_date(payload.get("date"))
        mode = payload.get("mode") or self._policies.default_mode
        mode_conf = self._policies.modes.get(mode, self._policies.modes[self._policies.default_mode])

        start_dt = datetime.combine(target_date, time(hour=self._policies.workday_start_hour, tzinfo=timezone.utc))
        end_dt = datetime.combine(target_date, time(hour=self._policies.workday_end_hour, tzinfo=timezone.utc))

        cal_events = self._calendar.search(start_dt, end_dt).get("events", [])
        tasks = self._tasks.list(status="pending")["tasks"]

        fixed_blocks = [self._event_block(event) for event in cal_events]
        prep_blocks = self._prep_blocks(cal_events, mode_conf.prep_buffer_min)
        blocks = fixed_blocks + prep_blocks

        focus_blocks = self._focus_blocks(tasks, blocks, start_dt, end_dt, mode_conf.focus_minutes)
        blocks.extend(focus_blocks)
        blocks.sort(key=lambda block: block.start)

        reminders = self._reminders_for_events(cal_events, mode_conf.prep_buffer_min)

        LOGGER.info(
            "planner.plan_day",
            date=target_date.isoformat(),
            mode=mode,
            scheduled_blocks=len(blocks),
            reminders=len(reminders),
        )
        return {
            "date": target_date.isoformat(),
            "mode": mode,
            "blocks": [block.to_dict() for block in blocks],
            "reminders": reminders,
        }
    @staticmethod
    def _parse_date(value: str | None) -> date:
        if value:
            return datetime.fromisoformat(value).date()
        return datetime.now(timezone.utc).date()

    @staticmethod
    def _event_block(event: dict[str, Any]) -> PlanBlock:
        start = datetime.fromisoformat(event["start"])
        end = datetime.fromisoformat(event["end"])
        label = event.get("title") or "Calendar event"
        return PlanBlock(
            type="event",
            start=start,
            end=end,
            label=label,
            extra={
                "id": event.get("id"),
                "attendees": event.get("attendees"),
                "location": event.get("location"),
            },
        )

    def _prep_blocks(self, events: list[dict[str, Any]], minutes: int) -> list[PlanBlock]:
        prep_blocks: list[PlanBlock] = []
        for event in events:
            start = datetime.fromisoformat(event["start"])
            prep_start = start - timedelta(minutes=minutes)
            prep_end = start
            if prep_start >= start.replace(hour=self._policies.workday_start_hour, minute=0, second=0, microsecond=0):
                prep_blocks.append(
                    PlanBlock(
                        type="prep",
                        start=prep_start,
                        end=prep_end,
                        label=f"Prep: {event.get('title')}",
                        extra={"event_id": event.get("id")},
                    )
                )
        return prep_blocks

    def _focus_blocks(
        self,
        tasks: list[dict[str, Any]],
        existing_blocks: list[PlanBlock],
        day_start: datetime,
        day_end: datetime,
        focus_minutes: int,
    ) -> list[PlanBlock]:
        occupied = sorted(existing_blocks, key=lambda block: block.start)
        focus_blocks: list[PlanBlock] = []
        cursor = day_start
        task_iter = iter(tasks)

        for block in occupied:
            while cursor + timedelta(minutes=focus_minutes) <= block.start:
                task = next(task_iter, None)
                if task is None:
                    return focus_blocks
                focus_end = cursor + timedelta(minutes=focus_minutes)
                focus_blocks.append(
                    PlanBlock(
                        type="focus",
                        start=cursor,
                        end=focus_end,
                        label=f"Focus: {task['title']}",
                        extra={"task_id": task["id"], "list": task.get("list_name")},
                    )
                )
                cursor = focus_end + timedelta(minutes=self._policies.microbreak_minutes)
            cursor = max(cursor, block.end)

        while cursor + timedelta(minutes=focus_minutes) <= day_end:
            task = next(task_iter, None)
            if task is None:
                break
            focus_end = cursor + timedelta(minutes=focus_minutes)
            focus_blocks.append(
                PlanBlock(
                    type="focus",
                    start=cursor,
                    end=focus_end,
                    label=f"Focus: {task['title']}",
                    extra={"task_id": task["id"], "list": task.get("list_name")},
                )
            )
            cursor = focus_end + timedelta(minutes=self._policies.microbreak_minutes)
        return focus_blocks

    def _reminders_for_events(self, events: list[dict[str, Any]], prep_minutes: int) -> list[dict[str, Any]]:
        reminders: list[dict[str, Any]] = []
        for event in events:
            start = datetime.fromisoformat(event["start"])
            end = datetime.fromisoformat(event["end"])
            prep_time = reminder_tools.meeting_prep(start, prep_minutes)
            reminders.append(
                {
                    "type": "prep",
                    "title": f"Prep for {event.get('title')}",
                    "when": prep_time.isoformat(),
                    "event_id": event.get("id"),
                }
            )
            follow_time = reminder_tools.follow_up(end, self._policies.followup_minutes)
            reminders.append(
                {
                    "type": "follow_up",
                    "title": f"Follow up: {event.get('title')}",
                    "when": follow_time.isoformat(),
                    "event_id": event.get("id"),
                }
            )
        return reminders


_PLANNER: DayPlanner | None = None


def get_planner() -> DayPlanner:
    global _PLANNER
    if _PLANNER is None:
        _PLANNER = DayPlanner()
    return _PLANNER


def plan_day(payload: dict[str, Any]) -> dict[str, Any]:
    return get_planner().plan_day(payload)


__all__ = ["DayPlanner", "PlanBlock", "plan_day", "get_planner"]
