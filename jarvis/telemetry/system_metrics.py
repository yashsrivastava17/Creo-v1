from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import psutil

from jarvis.telemetry.logging import get_logger


@dataclass
class SystemSample:
    cpu_percent: float
    mem_percent: float
    rss_mb: float


class MetricsSink(Protocol):
    async def publish_state(self, state: str, payload: dict) -> None: ...


class ResourceMonitor:
    def __init__(self, interval_seconds: int, ui_bridge: MetricsSink | None = None) -> None:
        self._interval = max(5, interval_seconds)
        self._ui_bridge = ui_bridge
        self._logger = get_logger(__name__)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="resource-monitor")
        self._logger.info("resource.monitor.started", interval=self._interval)

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._logger.info("resource.monitor.stopped")

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                sample = self._sample()
                self._logger.info(
                    "resource.monitor.sample",
                    cpu_percent=sample.cpu_percent,
                    mem_percent=sample.mem_percent,
                    rss_mb=sample.rss_mb,
                )
                if self._ui_bridge:
                    await self._ui_bridge.publish_state(
                        "RESOURCE",
                        {
                            "cpu_percent": sample.cpu_percent,
                            "mem_percent": sample.mem_percent,
                            "rss_mb": sample.rss_mb,
                        },
                    )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:  # pragma: no cover
            pass

    def _sample(self) -> SystemSample:
        process = psutil.Process()
        cpu = process.cpu_percent(interval=None)
        mem = process.memory_percent()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        if cpu == 0.0:
            cpu = psutil.cpu_percent(interval=0.1)
        return SystemSample(cpu_percent=cpu, mem_percent=mem, rss_mb=rss_mb)
