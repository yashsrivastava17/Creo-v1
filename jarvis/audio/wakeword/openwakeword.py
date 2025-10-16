from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

try:
    import openwakeword  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openwakeword = None

import sounddevice as sd

from jarvis.audio.wakeword.base import WakeWordEngine
from jarvis.orchestrator.events import WakeWordHit
from jarvis.telemetry.logging import get_logger


class OpenWakeWordEngine(WakeWordEngine):
    def __init__(self, keyword: str, sensitivity: float = 0.5) -> None:
        if openwakeword is None:
            raise RuntimeError("openwakeword is not installed.")
        self._keyword = keyword
        self._detector = openwakeword.Model(wakewords=[keyword], sensitivity=sensitivity)
        self._logger = get_logger(__name__)
        self._stream = sd.InputStream(
            samplerate=self._detector.sample_rate,
            channels=1,
            blocksize=self._detector.frame_length,
            dtype="int16",
        )
        self._running = False

    async def run(self) -> AsyncIterator[WakeWordHit]:
        loop = asyncio.get_running_loop()
        self._stream.start()
        self._running = True
        self._logger.info("wakeword.openwakeword.started", keyword=self._keyword)
        try:
            while self._running:
                frame = await loop.run_in_executor(None, self._stream.read, self._detector.frame_length)
                pcm = frame[0].flatten().astype(np.int16)
                scores = self._detector.predict(np.expand_dims(pcm, axis=0))
                if scores[0] > 0.5:
                    ts = time.time()
                    hit = WakeWordHit(ts=ts, keyword=self._keyword, confidence=float(scores[0]), buffer_ref_ms=4000)
                    self._logger.info("wakeword.hit", keyword=self._keyword, confidence=float(scores[0]))
                    yield hit
        finally:
            self._stream.stop()
            self._stream.close()

    async def close(self) -> None:
        self._running = False
        self._detector.reset()

