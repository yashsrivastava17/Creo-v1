from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from jarvis.orchestrator.events import TranscriptChunk


class TranscriptionEngine(ABC):
    @abstractmethod
    async def enqueue_audio(self, pcm: bytes, ts: float, vad: bool | None = None, force: bool = False) -> None:
        """Add audio data for transcription."""

    @abstractmethod
    async def stream(self) -> AsyncIterator[TranscriptChunk]:
        """Yield transcript chunks as they arrive."""

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
