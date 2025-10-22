from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.telemetry.logging import get_logger

LOGGER = get_logger(__name__)
CAPTURE_PATH = Path("/tmp/jarvis_ss.png")


@dataclass(slots=True)
class ScreenReading:
    text_blocks: list[str]
    active_app: str | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"text_blocks": self.text_blocks, "active_app": self.active_app, "meta": self.meta}


class SensorSuite:
    def __init__(self) -> None:
        self._work_minutes = 0

    def screen_ocr(self) -> dict[str, Any]:
        if self._capture_screen():
            LOGGER.info("sensors.screen.capture", path=str(CAPTURE_PATH))
        else:
            LOGGER.info("sensors.screen.capture_skipped")
        # TODO: Integrate Vision / Tesseract OCR and active app lookup.
        reading = ScreenReading(
            text_blocks=[],
            active_app=None,
            meta={
                "note": "Stub response – wire macOS Vision OCR for production.",
                "path": str(CAPTURE_PATH),
            },
        )
        return reading.to_dict()

    def activity_ping(self) -> dict[str, Any]:
        self._work_minutes += 1
        LOGGER.info("sensors.activity_ping", total_minutes=self._work_minutes)
        suggestion = None
        if self._work_minutes % 55 == 0:
            suggestion = "Time for a 5 minute microbreak."
        return {"work_minutes": self._work_minutes, "suggestion": suggestion}

    @staticmethod
    def _capture_screen() -> bool:
        if os.name != "posix":
            return False
        try:
            subprocess.run(["screencapture", "-x", str(CAPTURE_PATH)], check=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.SubprocessError, PermissionError):
            return False


SENSORS = SensorSuite()


def screen_ocr() -> dict[str, Any]:
    return SENSORS.screen_ocr()


def activity_ping() -> dict[str, Any]:
    return SENSORS.activity_ping()


__all__ = ["SensorSuite", "screen_ocr", "activity_ping", "SENSORS"]
