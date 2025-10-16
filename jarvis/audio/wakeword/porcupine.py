from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

try:
    import pvporcupine  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pvporcupine = None

import sounddevice as sd

from jarvis.audio.wakeword.base import WakeWordEngine
from jarvis.orchestrator.events import WakeWordHit
from jarvis.telemetry.logging import get_logger


class PorcupineWakeWordEngine(WakeWordEngine):
    def __init__(
        self,
        keyword_path: str,
        model_path: str | None = None,
        sensitivity: float = 0.6,
        access_key: str | None = None,
    ) -> None:
        if pvporcupine is None:
            raise RuntimeError("pvporcupine is not installed; install picovoice porcupine bindings.")
        if access_key is None:
            raise RuntimeError("Porcupine access key not provided. Set PORCUPINE_ACCESS_KEY in your .env.")
        self._keyword_path = keyword_path
        self._model_path = model_path
        self._sensitivity = sensitivity
        self._porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_path],
            model_path=model_path,
            sensitivities=[sensitivity],
        )
        self._logger = get_logger(__name__)
        self._stream = sd.InputStream(
            samplerate=self._porcupine.sample_rate,
            channels=1,
            blocksize=self._porcupine.frame_length,
            dtype="int16",
        )
        self._running = False
        self._closed = asyncio.Event()

    async def run(self) -> AsyncIterator[WakeWordHit]:
        self._stream.start()
        self._running = True
        loop = asyncio.get_running_loop()
        self._logger.info("wakeword.porcupine.started", keyword_path=self._keyword_path)
        try:
            while self._running:
                frame = await loop.run_in_executor(None, self._stream.read, self._porcupine.frame_length)
                pcm = frame[0].flatten().astype(np.int16)
                result = self._porcupine.process(pcm)
                if result >= 0:
                    ts = time.time()
                    hit = WakeWordHit(ts=ts, keyword="porcupine", confidence=1.0, buffer_ref_ms=4000)
                    self._logger.info("wakeword.hit", keyword=hit.keyword, ts=ts)
                    yield hit
        finally:
            self._stream.stop()
            self._stream.close()
            self._closed.set()

    async def close(self) -> None:
        if not self._running:
            if not self._closed.is_set():
                self._closed.set()
            self._porcupine.delete()
            return
        self._running = False
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._stream.abort)
        except Exception:
            pass
        await self._closed.wait()
        self._porcupine.delete()
