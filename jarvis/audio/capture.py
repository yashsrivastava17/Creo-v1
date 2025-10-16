from __future__ import annotations

import asyncio
import queue
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
import sounddevice as sd

from jarvis.audio.ring_buffer import RingBuffer
from jarvis.orchestrator.events import AudioFrame
from jarvis.telemetry.logging import get_logger


class AudioCapture:
    def __init__(
        self,
        samplerate: int = 16_000,
        channels: int = 1,
        frame_ms: int = 30,
        vad_detector: Callable[[np.ndarray], bool] | None = None,
        ring_seconds: float = 12.0,
        energy_threshold: float = 500.0,
        device: str | int | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(self.samplerate * self.frame_ms / 1000)
        self.ring_buffer = RingBuffer(ring_seconds, samplerate, self.frame_samples)
        self.vad_detector = vad_detector
        self.energy_threshold = energy_threshold
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._logger = get_logger(__name__)
        self._stream: sd.InputStream | None = None
        self._task: asyncio.Task[Any] | None = None
        self._device = device

    async def __aenter__(self) -> "AudioCapture":
        self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def start(self) -> None:
        if self._stream:
            return

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            if status:
                self._logger.warning("audio.capture.status", status=str(status))
            pcm = (indata.copy() * (2**15 - 1)).astype(np.int16).tobytes()
            self._queue.put_nowait(pcm)
            self.ring_buffer.push(pcm)

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.frame_samples,
            dtype="float32",
            callback=callback,
            device=self._device,
        )
        self._stream.start()
        self._logger.info(
            "audio.capture.started",
            samplerate=self.samplerate,
            frame_ms=self.frame_ms,
            energy_threshold=self.energy_threshold,
            device=self._device,
        )

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue.put_nowait(None)  # type: ignore[arg-type]
        self._logger.info("audio.capture.stopped")

    async def frames(self) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        while True:
            pcm = await loop.run_in_executor(None, self._queue.get)
            if pcm is None:
                break
            np_frame = np.frombuffer(pcm, dtype=np.int16)
            energy = float(np.abs(np_frame).mean())
            vad = self.vad_detector(np_frame) if self.vad_detector else self._default_vad(np_frame, energy)
            yield AudioFrame(ts=time.time(), pcm16le=pcm, vad=vad, energy=energy)

    def last_audio(self, seconds: float) -> bytes:
        frames = self.ring_buffer.snapshot(seconds)
        return b"".join(frames)

    def _default_vad(self, frame: np.ndarray, energy: float | None = None) -> bool:
        energy_val = float(energy) if energy is not None else float(np.abs(frame).mean())
        self._logger.debug("audio.energy", energy=energy_val)
        return energy_val > self.energy_threshold
