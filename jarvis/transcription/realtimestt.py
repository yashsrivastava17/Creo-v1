from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets

from jarvis.orchestrator.events import TranscriptChunk
from jarvis.telemetry.logging import get_logger


class RealTimeSTTClient:
    def __init__(self, base_url: str, auth_token: str | None = None) -> None:
        self._ws_url = base_url.rstrip("/") + "/ws"
        self._auth_token = auth_token
        self._logger = get_logger(__name__)

    async def audio_to_text(self, pcm16le: bytes, sample_rate: int) -> TranscriptChunk | None:
        """Fire-and-forget helper for synchronous chunks. Provided for testing."""
        async with websockets.connect(self._ws_url, extra_headers=self._headers()) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "sample_rate": sample_rate,
                        "audio": pcm16le.hex(),
                    }
                )
            )
            await ws.send(json.dumps({"type": "flush"}))
            async for message in ws:
                payload = json.loads(message)
                if payload.get("type") == "transcript":
                    return TranscriptChunk(
                        ts=payload.get("ts", 0.0),
                        text=payload["text"],
                        range_ms=(payload.get("start_ms", 0), payload.get("end_ms", 0)),
                        is_final=payload.get("is_final", False),
                    )
        return None

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers


class RealTimeSTTStream:
    def __init__(
        self,
        client: RealTimeSTTClient,
        sample_rate: int,
    ) -> None:
        self.client = client
        self._queue: asyncio.Queue[tuple[bytes | None, float, bool, bool]] = asyncio.Queue(maxsize=256)
        self._sample_rate = sample_rate
        self._logger = get_logger(__name__)
        self._stop_event = asyncio.Event()

    async def enqueue_audio(self, pcm: bytes, ts: float, vad: bool | None = None, force: bool = False) -> None:
        await self._queue.put((pcm, ts, bool(vad), force))

    async def stream(self) -> AsyncIterator[TranscriptChunk]:
        async with websockets.connect(self.client._ws_url, extra_headers=self.client._headers()) as ws:
            consumer = asyncio.create_task(self._consume_remote(ws))
            producer = asyncio.create_task(self._produce_audio(ws))
            try:
                async for message in ws:
                    payload = json.loads(message)
                    if payload.get("type") != "transcript":
                        continue
                    chunk = TranscriptChunk(
                        ts=payload.get("ts", 0.0),
                        text=payload["text"],
                        range_ms=(payload.get("start_ms", 0), payload.get("end_ms", 0)),
                        is_final=payload.get("is_final", False),
                    )
                    yield chunk
            finally:
                self._stop_event.set()
                producer.cancel()
                consumer.cancel()

    async def _produce_audio(self, ws: websockets.WebSocketClientProtocol) -> None:
        while not self._stop_event.is_set():
            pcm, ts, vad, force_flush = await self._queue.get()
            if pcm is None and not force_flush:
                break
            if force_flush:
                await ws.send(json.dumps({"type": "flush"}))
                continue
            await ws.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "sample_rate": self._sample_rate,
                        "audio": pcm.hex(),
                        "ts": ts,
                    }
                )
            )

    async def _consume_remote(self, ws: websockets.WebSocketClientProtocol) -> None:
        await ws.wait_closed()

    async def close(self) -> None:
        self._stop_event.set()
        await self._queue.put((None, 0.0, False, False))
