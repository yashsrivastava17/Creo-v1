from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from jarvis.orchestrator.policies import PlannerPolicies
from jarvis.telemetry.logging import get_logger
from jarvis.tools import calendar as calendar_tools
from jarvis.tools import reminders as reminder_tools
from jarvis.tools import tasks as task_tools

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class MeetingBrief:
    event: dict[str, Any]
    agenda: list[str]
    attendees: list[str]
    last_actions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "agenda": self.agenda,
            "attendees": self.attendees,
            "last_actions": self.last_actions,
        }


class MeetingTools:
    def __init__(self, policies: PlannerPolicies | None = None) -> None:
        self._policies = policies or PlannerPolicies()
        self._calendar = calendar_tools.get_calendar_client()
        self._tasks = task_tools.get_store()

    def brief(self, event_id: str) -> dict[str, Any]:
        event = self._get_event(event_id)
        agenda = self._extract_agenda(event)
        attendees = event.get("attendees") or []
        last_actions = self._related_tasks(event)
        payload = MeetingBrief(event=event, agenda=agenda, attendees=attendees, last_actions=last_actions).to_dict()
        LOGGER.info("meeting.brief", event_id=event_id)
        return payload

    def start_notes(self, event_id: str) -> dict[str, Any]:
        session_id = str(uuid4())
        LOGGER.info("meeting.start_notes", event_id=event_id, session_id=session_id)
        # TODO: Emit orchestrator event hook when note recorder is wired.
        return {"session_id": session_id, "event_id": event_id}

    def postprocess(self, event_id: str) -> dict[str, Any]:
        event = self._get_event(event_id)
        follow_up_task = self._tasks.add(
            title=f"Send recap for {event.get('title')}",
            list_name="Work",
        )["task"]
        follow_time = reminder_tools.follow_up(datetime.fromisoformat(event["end"]), self._policies.followup_minutes)
        reminder = reminder_tools.add(
            {
                "when_iso": follow_time.isoformat(),
                "title": f"Follow up after {event.get('title')}",
                "payload": {"event_id": event_id},
                "channel": "local",
            }
        )["reminder"]
        LOGGER.info("meeting.postprocess", event_id=event_id, task_id=follow_up_task["id"])
        return {"actions": [follow_up_task], "reminder": reminder}

    def _get_event(self, event_id: str) -> dict[str, Any]:
        event = self._calendar.get(event_id)
        if not event:
            raise ValueError(f"Unknown event id '{event_id}'")
        return event.to_dict() if hasattr(event, "to_dict") else event

    @staticmethod
    def _extract_agenda(event: dict[str, Any]) -> list[str]:
        notes = event.get("notes") or ""
        lines = [line.strip("•- ").strip() for line in notes.splitlines() if line.strip()]
        return lines[:5] if lines else ["Review objectives", "Capture action items"]

    def _related_tasks(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        tag = None
        for entry in event.get("tags", []):
            if entry.startswith("client:"):
                tag = entry.split(":", 1)[1]
                break
        tasks = self._tasks.list(list_names=None, status="pending")["tasks"]
        if not tag:
            return tasks[:3]
        return [task for task in tasks if tag.lower() in (task.get("title") or "").lower()][:3]


_MEETINGS: MeetingTools | None = None


def get_meeting_tools() -> MeetingTools:
    global _MEETINGS
    if _MEETINGS is None:
        _MEETINGS = MeetingTools()
    return _MEETINGS


def brief(event_id: str) -> dict[str, Any]:
    return get_meeting_tools().brief(event_id)


def start_notes(event_id: str) -> dict[str, Any]:
    return get_meeting_tools().start_notes(event_id)


def postprocess(event_id: str) -> dict[str, Any]:
    return get_meeting_tools().postprocess(event_id)


__all__ = ["MeetingTools", "brief", "start_notes", "postprocess", "get_meeting_tools"]
