from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence
from uuid import UUID


@dataclass(slots=True)
class AudioFrame:
    ts: float
    pcm16le: bytes
    vad: bool
    energy: float


@dataclass(slots=True)
class WakeWordHit:
    ts: float
    keyword: str
    confidence: float
    buffer_ref_ms: int


@dataclass(slots=True)
class TranscriptChunk:
    ts: float
    text: str
    range_ms: tuple[int, int]
    is_final: bool = False


@dataclass(slots=True)
class UserTurn:
    turn_id: UUID
    wake_word_ts: float
    prompt_text: str
    context: dict[str, Any]
    audio_refs: Sequence[str]
    tools_allowed: Sequence[str]


@dataclass(slots=True)
class ToolCall:
    turn_id: UUID
    tool: str
    args: dict[str, Any]


@dataclass(slots=True)
class AssistantMessage:
    turn_id: UUID
    text: str
    citations: list[dict[str, Any]]
    memory_writes: list[dict[str, Any]]
    tts_duration: float | None = None


State = Literal["IDLE", "LISTENING", "HOTWORD_HEARD", "COMPOSING", "SPEAKING", "MAINTENANCE", "RESOURCE", "PERSONA"]


__all__ = [
    "AudioFrame",
    "WakeWordHit",
    "TranscriptChunk",
    "UserTurn",
    "ToolCall",
    "AssistantMessage",
    "State",
]
