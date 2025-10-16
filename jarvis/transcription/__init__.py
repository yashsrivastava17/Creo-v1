try:
    from jarvis.transcription.cheetah import CheetahStream
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    CheetahStream = None  # type: ignore[assignment]

try:
    from jarvis.transcription.vosk import VoskStream
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    VoskStream = None  # type: ignore[assignment]

__all__ = ["CheetahStream", "VoskStream"]
