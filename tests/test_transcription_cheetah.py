import asyncio
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


class DummyCheetah:
    sample_rate = 16_000
    frame_length = 4

    def __init__(self) -> None:
        self._calls = 0

    def process(self, frame: list[float]) -> tuple[str, bool]:
        self._calls += 1
        if self._calls == 1:
            return ("hi", False)
        return ("", True)

    def flush(self) -> str:
        return "hello there"

    def delete(self) -> None:  # pragma: no cover - compatibility shim
        pass


@pytest.mark.anyio("asyncio")
async def test_cheetah_stream_yields_chunks(monkeypatch):
    dummy_module = types.ModuleType("pvcheetah")
    dummy_module.create = lambda **_: None  # patched below
    monkeypatch.setitem(sys.modules, "pvcheetah", dummy_module)

    from jarvis.transcription import cheetah

    dummy = DummyCheetah()
    monkeypatch.setattr(cheetah, "create", lambda **_: dummy)

    stream = cheetah.CheetahStream(access_key="key", model_path=None, endpoint_sec=0.6)

    chunks = []

    async def consume() -> None:
        async for chunk in stream.stream():
            chunks.append(chunk)
            if chunk.is_final:
                break

    consumer = asyncio.create_task(consume())

    pcm = (np.ones(dummy.frame_length, dtype=np.int16) * 1000).tobytes()
    await stream.enqueue_audio(pcm, ts=0.0, vad=True)
    await stream.enqueue_audio(b"", ts=0.1, vad=False, force=True)
    await stream.close()

    await consumer

    assert any(not chunk.is_final for chunk in chunks)
    assert any(chunk.is_final and chunk.text for chunk in chunks)
