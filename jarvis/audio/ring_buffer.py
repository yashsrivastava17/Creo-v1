from __future__ import annotations

import collections
import threading
import time
from typing import Deque


class RingBuffer:
    """Simple time-aware ring buffer for PCM16 frames."""

    def __init__(self, max_duration_sec: float, sample_rate: int, frame_size: int) -> None:
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._frames: Deque[tuple[float, bytes]] = collections.deque()
        self._lock = threading.Lock()

    def push(self, frame: bytes) -> None:
        ts = time.time()
        with self._lock:
            self._frames.append((ts, frame))
            self._prune_locked(ts)

    def snapshot(self, duration_sec: float) -> list[bytes]:
        cutoff = time.time() - duration_sec
        with self._lock:
            frames = [frame for ts, frame in self._frames if ts >= cutoff]
        return frames

    def _prune_locked(self, now: float) -> None:
        while self._frames:
            ts, _ = self._frames[0]
            if now - ts > self.max_duration_sec:
                self._frames.popleft()
            else:
                break

