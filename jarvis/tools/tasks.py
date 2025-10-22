from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from jarvis.telemetry.logging import get_logger

LOGGER = get_logger(__name__)
DEFAULT_DB = Path(__file__).resolve().parent / "_tasks.db"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class TasksStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path or DEFAULT_DB)
        self._ensure_schema()

    def sync_from_apple(self, list_names: list[str] | None = None) -> dict[str, Any]:
        LOGGER.info("tasks.sync.stub", lists=list_names)
        mirrored = self._stub_fetch(list_names)
        with self._connection() as conn:
            for task in mirrored:
                conn.execute(
                    """
                    INSERT INTO tasks (id, title, est_min, deadline, priority, list_name, status, created_at, updated_at)
                    VALUES (:id, :title, :est_min, :deadline, :priority, :list_name, :status, :created_at, :updated_at)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        est_min=excluded.est_min,
                        deadline=excluded.deadline,
                        priority=excluded.priority,
                        list_name=excluded.list_name,
                        status=excluded.status,
                        updated_at=excluded.updated_at
                    """,
                    task,
                )
        return {"tasks": self._fetch(list_names=list_names), "source": "stub"}

    def add(self, title: str, est_min: int | None = None, list_name: str | None = None) -> dict[str, Any]:
        task = {
            "id": str(uuid4()),
            "title": title,
            "est_min": est_min,
            "deadline": None,
            "priority": None,
            "list_name": list_name,
            "status": "pending",
            "created_at": _iso(datetime.now(timezone.utc)),
            "updated_at": _iso(datetime.now(timezone.utc)),
        }
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO tasks (id, title, est_min, deadline, priority, list_name, status, created_at, updated_at)
                VALUES (:id, :title, :est_min, :deadline, :priority, :list_name, :status, :created_at, :updated_at)
                """,
                task,
            )
        LOGGER.info("tasks.add", task=task)
        return {"task": task}

    def complete(self, task_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown task id '{task_id}'")
            updated = _iso(datetime.now(timezone.utc))
            conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                ("completed", updated, task_id),
            )
        payload = dict(row)
        payload.update({"status": "completed", "updated_at": updated})
        LOGGER.info("tasks.complete", task_id=task_id)
        return {"task": self._normalize(payload)}

    def list(self, list_names: list[str] | None = None, status: str | None = None) -> dict[str, Any]:
        return {"tasks": self._fetch(list_names=list_names, status=status)}

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    est_min INTEGER,
                    deadline TEXT,
                    priority TEXT,
                    list_name TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _fetch(self, list_names: list[str] | None = None, status: str | None = None) -> list[dict[str, Any]]:
        filters: list[str] = []
        params: list[Any] = []
        if list_names:
            placeholders = ",".join("?" for _ in list_names)
            filters.append(f"list_name IN ({placeholders})")
            params.extend(list_names)
        if status:
            filters.append("status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"SELECT * FROM tasks {clause} ORDER BY COALESCE(deadline, ''), created_at"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._normalize(dict(row)) for row in rows]

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        if row.get("deadline"):
            row["deadline"] = datetime.fromisoformat(row["deadline"]).isoformat()
        row["created_at"] = datetime.fromisoformat(row["created_at"]).isoformat()
        row["updated_at"] = datetime.fromisoformat(row["updated_at"]).isoformat()
        return row

    def _stub_fetch(self, list_names: list[str] | None) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        sample = [
            {
                "id": "apple://reminder/sample-1",
                "title": "Follow up with Avi",
                "est_min": None,
                "deadline": _iso(now.replace(hour=21, minute=0)),
                "priority": "medium",
                "list_name": "Personal",
                "status": "pending",
                "created_at": _iso(now),
                "updated_at": _iso(now),
            },
            {
                "id": "apple://reminder/sample-2",
                "title": "Prep deck for client sync",
                "est_min": 45,
                "deadline": None,
                "priority": "high",
                "list_name": "Work",
                "status": "pending",
                "created_at": _iso(now),
                "updated_at": _iso(now),
            },
        ]
        if list_names:
            sample = [task for task in sample if task["list_name"] in list_names]
        return sample


_TASKS_STORE: TasksStore | None = None


def get_store() -> TasksStore:
    global _TASKS_STORE
    if _TASKS_STORE is None:
        _TASKS_STORE = TasksStore()
    return _TASKS_STORE


def sync_from_apple(list_names: list[str] | None = None) -> dict[str, Any]:
    return get_store().sync_from_apple(list_names)


def add(title: str, est_min: int | None = None, list_name: str | None = None) -> dict[str, Any]:
    return get_store().add(title, est_min=est_min, list_name=list_name)


def complete(task_id: str) -> dict[str, Any]:
    return get_store().complete(task_id)


def list_tasks(list_names: list[str] | None = None, status: str | None = None) -> dict[str, Any]:
    return get_store().list(list_names=list_names, status=status)


__all__ = ["TasksStore", "sync_from_apple", "add", "complete", "list_tasks", "get_store"]
