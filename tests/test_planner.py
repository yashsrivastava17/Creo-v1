from __future__ import annotations

from datetime import datetime, timezone

from jarvis.orchestrator.policies import PlannerPolicies
from jarvis.tools import tasks as tasks_tools
from jarvis.tools.planner import DayPlanner


def test_plan_day_includes_focus_and_prep(monkeypatch, tmp_path) -> None:
    store = tasks_tools.TasksStore(db_path=tmp_path / "tasks.db")
    monkeypatch.setattr(tasks_tools, "get_store", lambda: store)
    store.add("Draft weekly update", est_min=30, list_name="Work")

    planner = DayPlanner(policies=PlannerPolicies())
    today_iso = datetime.now(timezone.utc).date().isoformat()
    result = planner.plan_day({"date": today_iso, "mode": "balanced"})

    block_types = {block["type"] for block in result["blocks"]}
    assert "prep" in block_types
    assert "focus" in block_types

    reminder_types = {reminder["type"] for reminder in result["reminders"]}
    assert {"prep", "follow_up"}.issubset(reminder_types)
