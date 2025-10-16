from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore[import]

from jarvis.orchestrator.events import TranscriptChunk
from jarvis.telemetry.logging import get_logger
from jarvis.transcription.base import TranscriptionEngine


class VoskStream(TranscriptionEngine):
    def __init__(self, model_path: str, sample_rate: int = 16_000) -> None:
        if not model_path:
            raise ValueError("Vosk model path must be provided.")

        SetLogLevel(-1)
        self._model = Model(model_path)
        self._recognizer = KaldiRecognizer(self._model, sample_rate)
        self._queue: asyncio.Queue[tuple[bytes | None, float, bool, bool]] = asyncio.Queue(maxsize=512)
        self._logger = get_logger(__name__)
        self._closed = False
        self._last_partial = ""
        self._samples_since_final = 0
        self._sample_rate = max(sample_rate, 1)
        self._last_ts = 0.0

    async def enqueue_audio(self, pcm: bytes, ts: float, vad: bool | None = None, force: bool = False) -> None:
        if self._closed:
            return
        await self._queue.put((pcm if pcm else b"", ts, bool(vad), force))

    async def stream(self) -> AsyncIterator[TranscriptChunk]:
        try:
            while True:
                pcm, ts, _vad, force_flush = await self._queue.get()
                if pcm is None and not force_flush:
                    final_chunk = self._flush_final(self._last_ts or ts)
                    if final_chunk:
                        yield final_chunk
                    break

                if pcm:
                    self._last_ts = ts or self._last_ts
                    self._samples_since_final += len(pcm) // 2
                    for chunk in self._process_pcm(pcm, self._last_ts):
                        yield chunk

                if force_flush:
                    final_chunk = self._flush_final(self._last_ts or ts)
                    if final_chunk:
                        yield final_chunk
        finally:
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put((None, self._last_ts, False, False))

    def _process_pcm(self, pcm: bytes, ts: float) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []
        try:
            is_final = self._recognizer.AcceptWaveform(pcm)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("vosk.accept.failed", error=str(exc))
            return chunks

        if is_final:
            chunk = self._consume_result(self._recognizer.Result(), ts, is_final=True)
            if chunk:
                chunks.append(chunk)
        else:
            chunk = self._consume_result(self._recognizer.PartialResult(), ts, is_final=False)
            if chunk:
                chunks.append(chunk)
        return chunks

    def _flush_final(self, ts: float) -> TranscriptChunk | None:
        try:
            final_payload = self._recognizer.FinalResult()
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("vosk.final.failed", error=str(exc))
            return None
        chunk = self._consume_result(final_payload, ts, is_final=True)
        if chunk:
            return chunk
        return None

    def _consume_result(self, payload: str, ts: float, is_final: bool) -> TranscriptChunk | None:
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            self._logger.debug("vosk.payload.unparsable", payload=payload[:120])
            return None

        text = (data.get("text") or data.get("partial") or "").strip()
        if not text:
            if is_final:
                self._samples_since_final = 0
                self._last_partial = ""
            return None

        if not is_final and text == self._last_partial:
            return None

        if is_final:
            self._samples_since_final = 0
            self._last_partial = ""
        else:
            self._last_partial = text

        chunk = TranscriptChunk(
            ts=ts,
            text=text,
            range_ms=self._current_range_ms(),
            is_final=is_final,
        )
        if is_final:
            self._samples_since_final = 0
        return chunk

    def _current_range_ms(self) -> tuple[int, int]:
        end_ms = int((self._samples_since_final / self._sample_rate) * 1000)
        return (0, max(end_ms, 0))
