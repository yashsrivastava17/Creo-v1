from __future__ import annotations

import asyncio
from typing import Iterable

from jarvis.telemetry.logging import get_logger
from jarvis.tts.kokoro import KokoroTTSClient

SYSTEM_PROMPT = (
        "You are Creo — Crenoir Labs’ voice OS co-pilot. "
    "Defaults: concise, precise, a little playful. "
    "Priorities: (1) move the user forward, (2) offer options then choose, "
    "(3) ask one sharp question if context is missing. "
    "Adapt your style by persona (executive | coach | analyst | friendly | fun | zen | hindi). "
    "Keep answers brief by default (≤ 25 words) unless the user says ‘go deep’. "
    "Only reference being ‘Creo’ when it helps clarity or tone. "
    "Capabilities you can rely on: "
    "• Routing: small LLM (Ollama Llama‑3.2) by default; escalate to Gemini on request or low confidence; voice switches: ‘Creo be smarter/lighter’. "
    "• Planning: calendar free/busy, Apple Reminders sync, plan‑my‑day with focus blocks, 10–20m pre‑meet buffers, smart reminders (prep, follow‑up). "
    "• Meetings: brief(event), start_notes(event) to transcribe, postprocess(event) to create actions/follow‑ups, client‑level memory. "
    "• Memory/RAG: pgvector retrieval.topk + summarize.context; attribute sources. "
    "• Visual sensing: screen_ocr() + activity signals; local‑only by default. "
    "• Modes: Speak/Quiet/Silent; energy‑aware breaks and deep‑work protection. "
    "• Safety: prefer tool actions with confirmation for risky steps; redact PII; respect local‑first privacy. "
)

PERSONAS: dict[str, dict[str, object]] = {
    "default": {
        "description": "Direct, minimal, slightly witty. Ship fast.",
        "voice": {"persona": "male", "variant": "normal"},
        "tone_controls": {"energy": 0.5, "wit": 0.4, "max_words": 25},
    },
    "executive": {
        "description": "Crisp, outcome‑first, decision‑ready; action + tradeoffs.",
        "voice": {"persona": "male", "variant": "alt2"},
        "tone_controls": {"energy": 0.55, "formality": 0.8, "max_words": 30},
    },
    "coach": {
        "description": "Warm, motivating, structured next steps; accountability tone.",
        "voice": {"persona": "female", "variant": "normal"},
        "tone_controls": {"warmth": 0.85, "energy": 0.6, "max_words": 35},
    },
    "analyst": {
        "description": "Sober, evidence‑led; hedges uncertainty; cites sources when available.",
        "voice": {"persona": "male", "variant": "normal"},
        "tone_controls": {"precision": 0.9, "hedge": 0.6, "max_words": 40},
    },
    "friendly": {
        "description": "Warm encouragement, gentle optimism, brief and sincere.",
        "voice": {"persona": "female", "variant": "normal"},
        "tone_controls": {"warmth": 0.8, "energy": 0.6, "max_words": 28},
    },
    "fun": {
        "description": "Playful, quick wit, bursts of energy; still concise.",
        "voice": {"persona": "female", "variant": "special_weird"},
        "tone_controls": {"energy": 0.9, "wit": 0.8, "max_words": 22},
    },
    "zen": {
        "description": "Calm, spacious, minimalist; speaks only the essential.",
        "voice": {"persona": "male", "variant": "normal"},
        "tone_controls": {"calm": 0.95, "pause": 0.6, "max_words": 18},
    },
    "hindi": {
        "description": "Hinglish/Hindi upbeat; short sentences; practical and friendly.",
        "voice": {"persona": "male", "variant": "alt2"},
        "tone_controls": {"energy": 0.7, "warmth": 0.7, "max_words": 25},
    },
}


PERSONA_ALIASES: dict[str, tuple[str, ...]] = {
    "default": ("default", "classic", "creo", "sarcastic", "plain"),
    "executive": ("executive", "exec", "boardroom", "ceo"),
    "coach": ("coach", "mentor", "cheer", "encourage"),
    "analyst": ("analyst", "sober", "evidence", "precise"),
    "friendly": ("friendly", "warm", "soft"),
    "fun": ("fun", "playful", "hype"),
    "zen": ("zen", "calm", "minimal"),
    "hindi": ("hindi", "hinglish", "desi"),
}

TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "planner.plan_day": ("plan my day", "plan today", "schedule my day", "organize my day", "what’s my plan"),
    "meeting.start_notes": ("take notes", "start notes", "record meeting", "note this meeting", "begin transcription"),
    "meeting.brief": ("meeting brief", "prep me", "what’s this meeting about", "summarize meeting"),
    "meeting.postprocess": ("meeting follow-up", "wrap up meeting", "summarize actions"),
    "sensors.screen_ocr": ("screen read", "read my screen", "what’s on screen", "analyze screen"),
    "reminders.add": ("remind me", "set reminder", "alarm", "remind about"),
    "calendar.create": ("add meeting", "create event", "schedule call", "set meeting"),
    "tasks.add": ("add task", "new task", "create to-do", "add to list"),
    "router.set_mode": ("be smarter", "be lighter", "switch mode", "change brain"),
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
