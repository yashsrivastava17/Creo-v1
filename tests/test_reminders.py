from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jarvis.tools import reminders


def test_schedule_and_cancel_reminder() -> None:
    scheduler = reminders.ReminderScheduler()
    when = datetime.now(timezone.utc) + timedelta(minutes=1)
    reminder = scheduler.add(when_iso=when.isoformat(), title="Ping", payload={})
    assert reminder.id in {item["id"] for item in scheduler.list_active()}
    assert scheduler.cancel(reminder.id) is True
    assert scheduler.cancel(reminder.id) is False


def test_meeting_helpers() -> None:
    start = datetime(2024, 4, 2, 16, 0, tzinfo=timezone.utc)
    assert reminders.meeting_prep(start, 15) == start - timedelta(minutes=15)
    assert reminders.follow_up(start, 10) == start + timedelta(minutes=10)
