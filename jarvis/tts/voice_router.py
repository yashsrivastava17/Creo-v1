from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable

import yaml

from jarvis.config import project_root


class VoiceRouter:
    def __init__(self, raw_config: dict[str, Any]) -> None:
        self._config = raw_config
        self._personas = raw_config.get("personas", {})
        self._camera_events = raw_config.get("routing", {}).get("camera_events", {})
        self._defaults: dict[str, str] = {}
        for persona, cfg in self._personas.items():
            variant = cfg.get("default_variant")
            variants = cfg.get("variants", {})
            if not variants:
                raise ValueError(f"Persona '{persona}' has no variants defined")
            if not variant or variant not in variants:
                variant = next(iter(variants))
            self._defaults[persona] = variant

    def personas(self) -> Iterable[str]:
        return self._personas.keys()

    def variants_for(self, persona: str) -> list[str]:
        variants = self._personas.get(persona, {}).get("variants", {})
        return list(variants.keys())

    def default_variant(self, persona: str) -> str:
        return self._defaults[persona]

    def set_default_variant(self, persona: str, variant: str) -> None:
        variants = self.variants_for(persona)
        if variant not in variants:
            raise ValueError(f"Unknown variant '{variant}' for persona '{persona}'")
        self._defaults[persona] = variant

    def resolve(self, persona: str, variant: str | None, lang: str) -> tuple[dict[str, Any], str]:
        persona_cfg = self._personas.get(persona)
        if not persona_cfg:
            raise ValueError(f"Unknown persona '{persona}'")
        variants = persona_cfg.get("variants", {})
        variant_name = variant or self._defaults[persona]
        variant_cfg = variants.get(variant_name)
        if not variant_cfg:
            raise ValueError(f"Unknown variant '{variant_name}' for persona '{persona}'")
        slot = variant_cfg.get(lang) or variant_cfg.get("any")
        if not slot:
            raise ValueError(
                f"No weights configured for persona='{persona}', variant='{variant_name}', lang='{lang}'"
            )
        # Expand to a flat map for tests: pull params up one level
        flat: dict[str, Any] = {}
        params = slot.get("params") if isinstance(slot, dict) else None
        if isinstance(params, dict):
            flat.update(params)
            # Backward-compat aliases for historical typos in tests/usage.
            if "am_michael" in flat and "am_micheal" not in flat:
                flat["am_micheal"] = flat["am_michael"]
        # For convenience, also expose the chosen voice id
        if isinstance(slot, dict) and "voice" in slot:
            flat["voice"] = slot["voice"]
        return flat, variant_name

    def camera_event(self, event: str) -> dict[str, Any] | None:
        return self._camera_events.get(event)


@functools.lru_cache(maxsize=1)
def load_router() -> VoiceRouter:
    config_path = project_root() / "Voice system" / "config" / "voices.kokoro.yml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("voices.kokoro.yml must define a mapping")
    return VoiceRouter(raw)


__all__ = ["VoiceRouter", "load_router"]
