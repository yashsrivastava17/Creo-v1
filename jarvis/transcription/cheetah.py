from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import numpy as np
from pvcheetah import create

from jarvis.orchestrator.events import TranscriptChunk
from jarvis.telemetry.logging import get_logger
from jarvis.transcription.base import TranscriptionEngine


class CheetahStream(TranscriptionEngine):
    def __init__(self, access_key: str | None = None, model_path: str | None = None, endpoint_sec: float | None = None, auto_punct: bool | None = None) -> None:
        # Load from environment if not provided
        if access_key is None:
            access_key = os.getenv("CHEETAH_ACCESS_KEY")
        if model_path is None:
            model_path = os.getenv("CHEETAH_MODEL_PATH") or None
        if endpoint_sec is None:
            try:
                endpoint_sec = float(os.getenv("CHEETAH_ENDPOINT_SEC", "0.6"))
            except ValueError:
                endpoint_sec = 0.6
        if auto_punct is None:
            auto_punct = os.getenv("CHEETAH_AUTO_PUNCT", "true").lower() == "true"
        if not access_key:
            raise ValueError("Picovoice Cheetah access key is required.")

        self._logger = get_logger(__name__)
        self._engine = create(
            access_key=access_key,
            model_path=model_path,
            endpoint_duration_sec=endpoint_sec,
            enable_automatic_punctuation=auto_punct,
        )
        self._logger.info(
            "transcription.cheetah.init",
            sample_rate=self._engine.sample_rate,
            frame_length=self._engine.frame_length,
            endpoint_sec=endpoint_sec,
        )
        self._queue: asyncio.Queue[tuple[bytes | None, float, bool, bool]] = asyncio.Queue(maxsize=512)
        self._closed = False
        self._pcm_buffer = bytearray()
        self._last_partial = ""
        self._samples_since_final = 0
        self._sample_rate = max(self._engine.sample_rate, 1)
        self._frame_length = self._engine.frame_length
        self._last_ts = 0.0

    async def enqueue_audio(self, pcm: bytes | np.ndarray, ts: float, vad: bool | None = None, force: bool = False) -> None:
        if self._closed:
            return
        # Accept float32 [-1,1] frames or raw int16 bytes
        if isinstance(pcm, np.ndarray):
            if pcm.dtype != np.float32:
                pcm = pcm.astype(np.float32)
            pcm_bytes = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        else:
            pcm_bytes = pcm if pcm else b""
        self._logger.debug(
            "transcription.cheetah.enqueue",
            bytes=len(pcm_bytes),
            force=force,
            vad=vad,
        )
        await self._queue.put((pcm_bytes, ts, bool(vad), force))

    async def stream(self) -> AsyncIterator[TranscriptChunk]:
        try:
            while True:
                pcm, ts, _vad, force_flush = await self._queue.get()
                if pcm is None and not force_flush:
                    # Final drain before exit.
                    chunk = self._flush_final(self._last_ts or ts)
                    if chunk:
                        yield chunk
                    break

                if pcm:
                    self._pcm_buffer.extend(pcm)
                    self._last_ts = ts or self._last_ts

                for chunk in self._process_ready_frames(self._last_ts or ts):
                    yield chunk

                if force_flush:
                    chunk = self._flush_final(self._last_ts or ts)
                    if chunk:
                        yield chunk
        finally:
            self._closed = True
            try:
                self._engine.delete()
            except Exception:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put((None, self._last_ts, False, False))

    def _process_ready_frames(self, ts: float) -> list[TranscriptChunk]:
        samples = np.frombuffer(self._pcm_buffer, dtype=np.int16)
        total_samples = len(samples)
        if total_samples < self._frame_length:
            self._logger.debug(
                "transcription.cheetah.buffering",
                buffered_samples=total_samples,
                required=self._frame_length,
            )
            return []

        usable_samples = (total_samples // self._frame_length) * self._frame_length
        if usable_samples == 0:
            return []

        ready_frames = samples[:usable_samples]
        self._pcm_buffer = bytearray(samples[usable_samples:].tobytes())

        emitted: list[TranscriptChunk] = []
        for offset in range(0, usable_samples, self._frame_length):
            frame = ready_frames[offset : offset + self._frame_length]
            int_frame = [int(sample) for sample in frame.tolist()]
            partial, endpoint = self._engine.process(int_frame)
            self._samples_since_final += len(frame)

            if partial:
                clean_partial = partial.strip()
                if clean_partial and clean_partial != self._last_partial:
                    self._logger.debug("transcription.cheetah.partial", text=clean_partial)
                    self._last_partial = clean_partial
                    emitted.append(
                        TranscriptChunk(
                            ts=ts,
                            text=clean_partial,
                            range_ms=self._current_range_ms(),
                            is_final=False,
                        )
                    )

            if endpoint:
                self._logger.debug("transcription.cheetah.endpoint", samples=self._samples_since_final)
                final_chunk = self._flush_final(ts)
                if final_chunk:
                    emitted.append(final_chunk)

        return emitted

    def _flush_final(self, ts: float) -> TranscriptChunk | None:
        final_text = self._engine.flush().strip()
        self._logger.debug("transcription.cheetah.flush", text=final_text)
        text = final_text or self._last_partial
        if not text:
            self._samples_since_final = 0
            self._last_partial = ""
            return None

        chunk = TranscriptChunk(
            ts=ts,
            text=text,
            range_ms=self._current_range_ms(),
            is_final=True,
        )
        self._samples_since_final = 0
        self._last_partial = ""
        return chunk

    def _current_range_ms(self) -> tuple[int, int]:
        end_ms = int((self._samples_since_final / self._sample_rate) * 1000)
        return (0, max(end_ms, 0))
