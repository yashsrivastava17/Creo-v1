from __future__ import annotations

import asyncio
from typing import Iterable

from jarvis.telemetry.logging import get_logger
from jarvis.tts.kokoro import KokoroTTSClient

SYSTEM_PROMPT = (
    "You are Creo, a concise, slightly sarcastic assistant built by Crenoir Labs. "
    "• Replies must be ≤ 10 words. "
    "• Maintain a dry, playful tone. "
    "• Always reference being ‘Creo’—the creation of Yash. "
    "• Focus on helping people bring ideas into existence. "
    "If you lack context, ask a short follow-up question."
)

PERSONAS: dict[str, dict[str, object]] = {
    "default": {
        "description": "Sarcastic helper; < 10 words; proud ‘creo’ built by Yash.",
        "voice": {"persona": "male", "variant": None},
        "tone_controls": {"sarcasm": 0.7, "energy": 0.5},
    },
    "friendly": {
        "description": "Warm encouragement, gentle optimism, brief and sincere.",
        "voice": {"persona": "female", "variant": None},
        "tone_controls": {"warmth": 0.8, "energy": 0.6},
    },
    "fun": {
        "description": "Playful, quick wit, bursts of energy, still concise.",
        "voice": {"persona": "female", "variant": "special_weird"},
        "tone_controls": {"energy": 0.85, "sarcasm": 0.4},
    },
}

PERSONA_ALIASES: dict[str, tuple[str, ...]] = {
    "default": ("default", "sarcastic", "classic", "creo"),
    "friendly": ("friendly", "warm", "soft"),
    "fun": ("fun", "playful", "hype"),
}

DEFAULT_PERSONA = "default"


class PersonaManager:
    def __init__(self, tts: KokoroTTSClient, initial: str = DEFAULT_PERSONA, language: str = "english") -> None:
        if initial not in PERSONAS:
            initial = DEFAULT_PERSONA
        self._tts = tts
        self._active = initial
        self._language = language
        self._voice_persona, self._voice_variant = self._resolve_voice_profile(initial)
        self._logger = get_logger(__name__)
        self._lock = asyncio.Lock()

    @property
    def active_name(self) -> str:
        return self._active

    @property
    def tts(self) -> KokoroTTSClient:
        return self._tts

    def active_profile(self) -> dict[str, object]:
        persona = PERSONAS[self._active]
        return {
            "name": self._active,
            "description": persona.get("description", ""),
            "voice_persona": self._voice_persona,
            "voice_variant": self._voice_variant,
            "tone_controls": persona.get("tone_controls", {}),
        }

    async def initialize(self) -> None:
        await self.activate(self._active, self._language)

    async def activate(self, persona_name: str, language: str | None = None) -> None:
        if persona_name not in PERSONAS:
            raise ValueError(f"unknown persona '{persona_name}'")

        async with self._lock:
            self._active = persona_name
            self._language = language or self._language
            self._voice_persona, self._voice_variant = self._resolve_voice_profile(self._active)
            await self._apply_voice()
            self._logger.info(
                "persona.activated",
                persona=self._active,
                voice_persona=self._voice_persona,
                voice_variant=self._voice_variant,
            )

    async def set_language(self, language: str) -> None:
        normalized = language or self._language
        async with self._lock:
            if normalized == self._language:
                return
            self._language = normalized
            await self._apply_voice()
            self._logger.info("persona.language.set", language=self._language)

    async def override_voice(self, voice_persona: str, variant: str | None) -> None:
        async with self._lock:
            self._voice_persona = voice_persona
            self._voice_variant = variant
            await self._tts.configure_voice(voice_persona, variant)
            self._logger.info(
                "persona.voice.override",
                voice_persona=self._voice_persona,
                voice_variant=self._voice_variant,
            )

    def match_command(self, transcript: str) -> str | None:
        lowered = transcript.lower()
        if "voice" not in lowered and "persona" not in lowered and "creo" not in lowered:
            return None
        for persona, aliases in PERSONA_ALIASES.items():
            if any(alias in lowered for alias in aliases):
                return persona
        return None

    def _resolve_voice_profile(self, persona_name: str) -> tuple[str, str | None]:
        persona = PERSONAS.get(persona_name, {})
        cfg = persona.get("voice", {}) if isinstance(persona.get("voice"), dict) else {}
        voice_persona = str(cfg.get("persona", "male"))
        variant = cfg.get("variant")
        variant_value = str(variant) if isinstance(variant, str) else None
        return voice_persona, variant_value

    async def _apply_voice(self) -> None:
        await self._tts.configure_voice(self._voice_persona, self._voice_variant)


__all__ = ["SYSTEM_PROMPT", "PERSONAS", "PERSONA_ALIASES", "PersonaManager", "DEFAULT_PERSONA"]
