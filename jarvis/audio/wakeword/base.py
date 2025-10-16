from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from jarvis.orchestrator.events import WakeWordHit


class WakeWordEngine(ABC):
    @abstractmethod
    async def run(self) -> AsyncIterator[WakeWordHit]:
        """Yield wake word hits as they are detected."""

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""

