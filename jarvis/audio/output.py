from __future__ import annotations

import asyncio
import io
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from jarvis.telemetry.logging import get_logger


class AudioOutputController:
    """Managed audio playback with cooperative interruption support."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._current_tag: Optional[str] = None
        self._current_task: Optional[asyncio.Task[None]] = None
        self._current_done: Optional[asyncio.Event] = None
        self._logger = get_logger(__name__)

    async def play_bytes(self, audio: bytes, tag: str, await_completion: bool = True) -> float:
        """Decode a byte buffer and play it via the shared controller."""
        if not audio:
            self._logger.warning("audio.output.empty_bytes", tag=tag)
            return 0.0
        try:
            with io.BytesIO(audio) as buffer:
                data, samplerate = sf.read(buffer, dtype="float32")
        except Exception as exc:  # pragma: no cover - defensive safety
            self._logger.error("audio.output.decode_failed", tag=tag, error=str(exc))
            return 0.0
        return await self.play_array(np.asarray(data), int(samplerate), tag, await_completion=await_completion)

    async def play_array(
        self,
        data: np.ndarray,
        samplerate: int,
        tag: str,
        await_completion: bool = True,
    ) -> float:
        """Play a numpy array of audio samples."""
        if samplerate <= 0 or data.size == 0:
            self._logger.warning("audio.output.invalid_payload", tag=tag, samplerate=samplerate, frames=int(data.size))
            return 0.0

        duration = data.shape[0] / float(samplerate)
        loop = asyncio.get_running_loop()
        done_event = asyncio.Event()

        async with self._lock:
            self._current_tag = tag
            self._current_done = done_event

            def _play() -> None:
                try:
                    sd.play(data, samplerate=samplerate, blocking=False)
                    sd.wait()
                except Exception as exc:  # pragma: no cover - defensive playback
                    self._logger.error("audio.output.play_error", tag=tag, error=str(exc))
                finally:
                    try:
                        sd.stop()
                    finally:
                        loop.call_soon_threadsafe(done_event.set)

            task = asyncio.create_task(asyncio.to_thread(_play))
            self._current_task = task

        if await_completion:
            await done_event.wait()
            await self._finalise_task(task)
        else:
            asyncio.create_task(self._finalise_task(task))
        return max(duration, 0.0)

    async def stop(self, tag: str | None = None) -> bool:
        """Stop current playback if tags match (or any playback when tag is None)."""
        async with self._lock:
            current_tag = self._current_tag
            done = self._current_done
        if current_tag is None:
            return False
        if tag is not None and current_tag != tag:
            return False
        sd.stop()
        if done:
            await done.wait()
        return True

    def current_tag(self) -> Optional[str]:
        return self._current_tag

    async def _finalise_task(self, task: asyncio.Task[None]) -> None:
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("audio.output.task_error", error=str(exc))
        finally:
            async with self._lock:
                if self._current_task is task:
                    self._current_task = None
                    self._current_tag = None
                    self._current_done = None


__all__ = ["AudioOutputController"]
