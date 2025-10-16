from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Optional

from jarvis.telemetry.logging import get_logger


class SelfMaintenanceScheduler:
    def __init__(self, ui_bridge, interval: timedelta | None = None) -> None:
        self._ui = ui_bridge
        self._interval = interval or timedelta(hours=4)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._logger = get_logger(__name__)

    async def start(self) -> None:
        if self._task:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="self-maintenance")
        self._logger.info("maintenance.scheduler.started", interval=str(self._interval))

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._logger.info("maintenance.scheduler.stopped")

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._interval.total_seconds())
                if self._stop.is_set():
                    break
                await self._ui.publish_state(
                    "MAINTENANCE",
                    {
                        "prompt": "I'd love feedback to improve. Would you like to review recent interactions or tune my preferences?",
                        "tags": ["rlhf", "maintenance"],
                    },
                )
        except asyncio.CancelledError:  # pragma: no cover
            pass
