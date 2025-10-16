from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from jarvis.memory import models
from jarvis.orchestrator.events import UserTurn
from jarvis.telemetry.logging import get_logger


class MemoryStore:
    def __init__(self, dsn: str, max_pool_size: int = 10) -> None:
        self._engine: AsyncEngine = create_async_engine(dsn, pool_size=max_pool_size, echo=False)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self._logger = get_logger(__name__)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.create_all)

    async def recall(self, user_turn: UserTurn | None) -> dict[str, Any]:
        # Placeholder recall logic; fetch profile + recent ephemeral facts.
        async with self._session_factory() as session:
            profile = (
                await session.execute(
                    select(models.MemoryFact).where(models.MemoryFact.user_id == 1, models.MemoryFact.kind == "profile")
                )
            ).scalars()
            ephemerals = (
                await session.execute(
                    select(models.MemoryFact)
                    .where(models.MemoryFact.user_id == 1, models.MemoryFact.kind == "ephemeral")
                    .order_by(models.MemoryFact.created_at.desc())
                    .limit(3)
                )
            ).scalars()
            return {
                "profile": [row.content for row in profile],
                "ephemeral": [row.content for row in ephemerals],
                "tools_allowed": ["web.search", "calendar.read"],
            }

    async def guard(self, memory_writes: list[dict]) -> list[dict]:
        # TODO: implement guard rails; placeholder passes through.
        return memory_writes

    async def write(self, writes: Sequence[dict]) -> None:
        if not writes:
            return
        async with self._session_factory() as session:
            for payload in writes:
                fact = models.MemoryFact(user_id=1, kind=payload.get("type", "fact"), content=payload.get("data", {}))
                session.add(fact)
            await session.commit()


class NullMemoryStore:
    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    async def init(self) -> None:  # pragma: no cover - trivial
        self._logger.warning("memory.null.init")

    async def recall(self, user_turn: UserTurn | None) -> dict[str, Any]:
        return {"profile": [], "ephemeral": [], "tools_allowed": []}

    async def guard(self, memory_writes: list[dict]) -> list[dict]:
        return []

    async def write(self, writes: Sequence[dict]) -> None:
        if writes:
            self._logger.warning("memory.null.write_ignored", writes=len(writes))
