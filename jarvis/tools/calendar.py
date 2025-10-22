from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from jarvis.telemetry.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CalendarEvent:
    id: str
    title: str
    start: datetime
    end: datetime
    location: str | None = None
    notes: str | None = None
    attendees: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start"] = self.start.isoformat()
        payload["end"] = self.end.isoformat()
        return payload


class AppleCalendarClient:
    """Mac-friendly calendar bridge.

    A lightweight in-memory stub stands in for the EventKit bridge until the native
    layer is wired up. Each method logs a TODO so we can swap in the real adapter
    without changing call sites.
    """

    def __init__(self) -> None:
        self._events: dict[str, CalendarEvent] = {}
        self._seed_demo_events()

    def freebusy(self, time_min: datetime, time_max: datetime) -> dict[str, Any]:
        LOGGER.info(
            "calendar.freebusy.stub",
            time_min=time_min.isoformat(),
            time_max=time_max.isoformat(),
        )
        busy = [
            {
                "start": event.start.isoformat(),
                "end": event.end.isoformat(),
                "title": event.title,
            }
            for event in self._iter_between(time_min, time_max)
        ]
        return {"busy": busy, "source": "stub", "todo": "Replace with EventKit bridge"}

    def search(self, time_min: datetime, time_max: datetime, query: str | None = None) -> dict[str, Any]:
        LOGGER.info(
            "calendar.search.stub",
            time_min=time_min.isoformat(),
            time_max=time_max.isoformat(),
            query=query,
        )
        events = [
            event.to_dict()
            for event in self._iter_between(time_min, time_max)
            if not query or self._matches(event, query)
        ]
        return {"events": events, "source": "stub", "todo": "Replace with EventKit bridge"}

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        LOGGER.info("calendar.create.stub", payload=payload)
        event = self._from_payload(payload, event_id=str(uuid4()))
        self._events[event.id] = event
        return {"event": event.to_dict(), "source": "stub", "todo": "Persist via EventKit"}

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = payload.get("id")
        if not event_id or event_id not in self._events:
            raise ValueError("Unknown calendar event id")
        LOGGER.info("calendar.update.stub", payload=payload)
        updated = self._from_payload(payload, event_id=event_id, original=self._events[event_id])
        self._events[event_id] = updated
        return {"event": updated.to_dict(), "source": "stub", "todo": "Persist via EventKit"}

    def get(self, event_id: str) -> CalendarEvent | None:
        return self._events.get(event_id)

    def _iter_between(self, time_min: datetime, time_max: datetime) -> Iterable[CalendarEvent]:
        for event in self._events.values():
            if event.end <= time_min or event.start >= time_max:
                continue
            yield event

    @staticmethod
    def _matches(event: CalendarEvent, query: str) -> bool:
        lower = query.lower()
        haystacks = filter(None, [event.title, event.notes, " ".join(event.tags)])
        return any(lower in field.lower() for field in haystacks)

    def _from_payload(
        self,
        payload: dict[str, Any],
        *,
        event_id: str,
        original: CalendarEvent | None = None,
    ) -> CalendarEvent:
        start = self._parse_dt(payload.get("start") or (original.start.isoformat() if original else None))
        end = self._parse_dt(payload.get("end") or (original.end.isoformat() if original else None))
        if start is None or end is None:
            raise ValueError("Event requires start and end timestamps")

        return CalendarEvent(
            id=event_id,
            title=payload.get("title") or (original.title if original else "Untitled"),
            start=start,
            end=end,
            location=payload.get("location", original.location if original else None),
            notes=payload.get("notes", original.notes if original else None),
            attendees=payload.get("attendees", original.attendees if original else []),
            tags=payload.get("tags", original.tags if original else []),
        )

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if value is None:
            return None
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _seed_demo_events(self) -> None:
        now = datetime.now(timezone.utc)
        sample = [
            CalendarEvent(
                id=str(uuid4()),
                title="Daily stand-up",
                start=now.replace(hour=15, minute=0, second=0, microsecond=0),
                end=now.replace(hour=15, minute=15, second=0, microsecond=0),
                attendees=["you@company.com"],
                tags=["team"],
            ),
            CalendarEvent(
                id=str(uuid4()),
                title="Project sync",
                start=now.replace(hour=17, minute=0, second=0, microsecond=0),
                end=now.replace(hour=18, minute=0, second=0, microsecond=0),
                attendees=["client@example.com"],
                notes="Review milestones and blockers.",
                tags=["client:acme"],
            ),
        ]
        for event in sample:
            self._events[event.id] = event


_CAL_CLIENT: AppleCalendarClient | None = None


def get_calendar_client() -> AppleCalendarClient:
    global _CAL_CLIENT
    if _CAL_CLIENT is None:
        _CAL_CLIENT = AppleCalendarClient()
    return _CAL_CLIENT


__all__ = ["CalendarEvent", "AppleCalendarClient", "get_calendar_client"]
