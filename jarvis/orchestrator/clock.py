from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from jarvis.telemetry.logging import get_logger

LOGGER = get_logger(__name__)


class Clock:
    """Asynchronous clock helper for background loops."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def sleep(self, seconds: float) -> None:
        LOGGER.debug("clock.sleep", seconds=seconds)
        await asyncio.sleep(seconds)

    async def ticks(self, interval: timedelta) -> AsyncIterator[datetime]:
        """Yield timestamps every `interval` seconds."""
        if interval.total_seconds() <= 0:
            raise ValueError("Interval must be positive")
        while True:
            await self.sleep(interval.total_seconds())
            yield self.now()


CLOCK = Clock()


def now() -> datetime:
    return CLOCK.now()


async def sleep(seconds: float) -> None:
    await CLOCK.sleep(seconds)


async def tick(interval_seconds: float) -> AsyncIterator[datetime]:
    async for ts in CLOCK.ticks(timedelta(seconds=interval_seconds)):
        yield ts


__all__ = ["Clock", "CLOCK", "now", "sleep", "tick"]
