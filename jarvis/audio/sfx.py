from __future__ import annotations

import asyncio
import datetime as dt
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from jarvis.audio.output import AudioOutputController
from jarvis.telemetry.logging import get_logger

EVENT_LABELS: dict[str, str] = {
    "wakeword": "Wakeword",
    "boot_sequence": "Boot Sequence",
    "small_error": "Small Error",
    "big_error": "Big Error",
    "interruption": "Interruption",
    "sleep_mode": "Sleep Mode",
    "exit": "Exit / Shutdown",
}

EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wakeword": ("wakeword", "wake_word", "heycreo", "hotword"),
    "boot_sequence": ("bootsequence", "boot_sequence", "boot", "startup"),
    "small_error": ("small_error", "warning", "minor", "soft_error"),
    "big_error": ("big_error", "fatal", "critical", "hard_error"),
    "interruption": ("interruption", "interrupt", "speechinterruption"),
    "sleep_mode": ("sleepmode", "sleep_mode", "sleep", "idle"),
    "exit": ("shutdown", "exit", "goodbye", "poweroff"),
}

ALLOWED_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


class SoundEffectManager:
    """Manage configured sound effects and playback routing."""

    def __init__(
        self,
        base_dir: Path,
        audio_output: AudioOutputController,
        config_path: Optional[Path] = None,
    ) -> None:
        self._base_dir = base_dir
        self._audio_output = audio_output
        self._config_path = config_path or base_dir / "sound_effects.yml"
        self._logger = get_logger(__name__)
        self._lock = asyncio.Lock()
        self._state: dict[str, dict[str, Any]] = {"events": {}}
        self._load_config()
        self._ensure_defaults()

    async def initialize(self) -> None:
        """No-op placeholder for interface symmetry."""
        return

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self._lock:
            file_names = [path.name for path in self._scan_files()]
            events = []
            for event, label in EVENT_LABELS.items():
                options = self._options_for(event, file_names)
                selected = self._state["events"].get(event, {}).get("selected")
                if selected and selected not in options and (self._base_dir / selected).exists():
                    options.append(selected)
                events.append(
                    {
                        "event": event,
                        "label": label,
                        "selected": selected,
                        "options": sorted(dict.fromkeys(options)),
                    }
                )
            return events

    async def select(self, event: str, filename: str) -> None:
        if event not in EVENT_LABELS:
            raise ValueError(f"Unknown sound effect event '{event}'")
        candidate = self._base_dir / filename
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        async with self._lock:
            self._state["events"].setdefault(event, {})["selected"] = filename
            self._write_config()
        self._logger.info("soundfx.selected", event=event, file=filename)

    async def add(self, event: str, original_name: str, data: bytes) -> str:
        if event not in EVENT_LABELS:
            raise ValueError(f"Unknown sound effect event '{event}'")
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported audio format '{suffix}'")
        slug = _slugify(Path(original_name).stem) or "clip"
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        base_name = f"{event}_{slug}_{timestamp}"
        dest = self._base_dir / f"{base_name}{suffix}"
        counter = 1
        self._base_dir.mkdir(parents=True, exist_ok=True)
        while dest.exists():
            dest = self._base_dir / f"{base_name}_{counter}{suffix}"
            counter += 1
        dest.write_bytes(data)
        self._logger.info("soundfx.uploaded", event=event, file=dest.name, bytes=len(data))
        return dest.name

    async def play(self, event: str, await_completion: bool = False) -> float:
        async with self._lock:
            selected = self._state["events"].get(event, {}).get("selected")
        if not selected:
            self._logger.debug("soundfx.play.no_selection", event=event)
            return 0.0
        path = self._base_dir / selected
        if not path.exists():
            self._logger.warning("soundfx.play.missing_file", event=event, file=selected)
            await self._mark_missing(event, selected)
            return 0.0
        try:
            audio = path.read_bytes()
        except Exception as exc:  # pragma: no cover - defensive read
            self._logger.error("soundfx.play.read_error", event=event, file=selected, error=str(exc))
            return 0.0
        tag = f"sfx:{event}"
        return await self._audio_output.play_bytes(audio, tag=tag, await_completion=await_completion)

    def _scan_files(self) -> list[Path]:
        if not self._base_dir.exists():
            self._base_dir.mkdir(parents=True, exist_ok=True)
            return []
        files = [
            path
            for path in sorted(self._base_dir.iterdir())
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
        ]
        return files

    def _options_for(self, event: str, file_names: list[str]) -> list[str]:
        keywords = EVENT_KEYWORDS.get(event, ())
        matched = [name for name in file_names if _matches_keywords(name, keywords)]
        if not matched:
            matched = list(file_names)
        return sorted(dict.fromkeys(matched))

    def _ensure_defaults(self) -> None:
        files = [path.name for path in self._scan_files()]
        changed = False
        for event in EVENT_LABELS:
            event_state = self._state["events"].setdefault(event, {"selected": None})
            selected = event_state.get("selected")
            if selected and selected not in files:
                event_state["selected"] = None
                changed = True
            if not event_state.get("selected"):
                candidates = self._options_for(event, files)
                if candidates:
                    event_state["selected"] = candidates[0]
                    changed = True
        if changed:
            self._write_config()

    def _load_config(self) -> None:
        if not self._config_path.exists():
            self._state = {"events": {}}
            return
        try:
            data = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive read
            self._logger.error("soundfx.config.load_failed", error=str(exc))
            data = {}
        events = data.get("events", {}) if isinstance(data, dict) else {}
        if not isinstance(events, dict):
            events = {}
        self._state = {"events": {}}
        for event in EVENT_LABELS:
            entry = events.get(event)
            if isinstance(entry, dict):
                self._state["events"][event] = {"selected": entry.get("selected")}
            else:
                self._state["events"][event] = {"selected": None}

    async def _mark_missing(self, event: str, filename: str) -> None:
        async with self._lock:
            stored = self._state["events"].get(event)
            if stored and stored.get("selected") == filename:
                stored["selected"] = None
                self._write_config()

    def _write_config(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self._state, handle, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - defensive write
            self._logger.error("soundfx.config.save_failed", error=str(exc))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_")


def _matches_keywords(name: str, keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return False
    slug = _slugify(Path(name).stem)
    return any(slug == key or slug.startswith(f"{key}_") or key in slug for key in keywords)


__all__ = ["SoundEffectManager", "EVENT_LABELS"]
