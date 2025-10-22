from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jarvis.telemetry.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_when(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass(slots=True)
class Reminder:
    id: str
    when: datetime
    title: str
    payload: dict[str, Any]
    channel: str = "local"
    created_at: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["when"] = self.when.isoformat()
        data["created_at"] = self.created_at.isoformat()
        return data


class ReminderScheduler:
    """Minimal in-process scheduler stub.

    TODO: Replace with APScheduler + macOS notification bridge for production.
    """

    def __init__(self) -> None:
        self._reminders: dict[str, Reminder] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def add(self, when_iso: str, title: str, payload: dict[str, Any], channel: str = "local") -> Reminder:
        when = _parse_when(when_iso)
        reminder = Reminder(id=str(uuid4()), when=when, title=title, payload=payload, channel=channel)
        with self._lock:
            self._reminders[reminder.id] = reminder
            self._arm_timer(reminder)
        LOGGER.info("reminders.add", reminder=reminder.to_dict())
        return reminder

    def cancel(self, reminder_id: str) -> bool:
        with self._lock:
            reminder = self._reminders.pop(reminder_id, None)
            timer = self._timers.pop(reminder_id, None)
        if timer:
            timer.cancel()
        if reminder:
            LOGGER.info("reminders.cancelled", reminder_id=reminder_id)
            return True
        return False

    def list_active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [reminder.to_dict() for reminder in self._reminders.values()]

    def _arm_timer(self, reminder: Reminder) -> None:
        delta = (reminder.when - datetime.now(timezone.utc)).total_seconds()
        if delta <= 0:
            LOGGER.warning("reminders.late_trigger", reminder=reminder.to_dict())
            return
        timer = threading.Timer(delta, self._trigger, args=(reminder.id,))
        timer.daemon = True
        self._timers[reminder.id] = timer
        timer.start()

    def _trigger(self, reminder_id: str) -> None:
        with self._lock:
            reminder = self._reminders.pop(reminder_id, None)
            self._timers.pop(reminder_id, None)
        if reminder is None:
            return
        LOGGER.info("reminders.fire", reminder=reminder.to_dict())
        # TODO: Bridge to Apple Reminders / local notification centre.


SCHEDULER = ReminderScheduler()


def add(request: dict[str, Any]) -> dict[str, Any]:
    reminder = SCHEDULER.add(
        when_iso=request["when_iso"],
        title=request.get("title", "Reminder"),
        payload=request.get("payload", {}),
        channel=request.get("channel", "local"),
    )
    return {"reminder": reminder.to_dict()}


def cancel(request: dict[str, Any]) -> dict[str, Any]:
    success = SCHEDULER.cancel(request["id"])
    return {"cancelled": success}


def meeting_prep(event_start: datetime, minutes_before: int = 15) -> datetime:
    return event_start - timedelta(minutes=minutes_before)


def follow_up(event_end: datetime, minutes_after: int = 10) -> datetime:
    return event_end + timedelta(minutes=minutes_after)


__all__ = [
    "ReminderScheduler",
    "add",
    "cancel",
    "meeting_prep",
    "follow_up",
    "SCHEDULER",
]
