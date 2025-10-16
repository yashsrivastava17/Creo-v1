from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

import httpx

from jarvis.audio.output import AudioOutputController
from jarvis.lang.script_detect import detect_lang_from_text
from jarvis.telemetry.logging import get_logger
from jarvis.tts.voice_router import VoiceRouter, load_router


class KokoroTTSClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        default_voice_params: Mapping[str, float | str] | None,
        router: VoiceRouter | None = None,
        audio_output: AudioOutputController | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._router = router or load_router()
        self._default_voice_params = dict(default_voice_params or {})
        self._current_persona: str | None = None
        self._current_variant: str | None = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
        self._audio_output = audio_output or AudioOutputController()
        self._logger = get_logger(__name__)

    async def configure_voice(self, persona: str, variant: str | None = None) -> None:
        """Set the active Kokoro persona + variant for subsequent calls."""
        self._current_persona = persona
        self._current_variant = variant

    def _build_request(
        self,
        text: str,
        voice_overrides: Mapping[str, float | str] | None,
        persona: str | None,
        variant: str | None,
    ) -> tuple[dict[str, object], str, str, str]:
        persona_key = persona or self._current_persona or next(iter(self._router.personas()))
        lang = detect_lang_from_text(text)
        weights, resolved_variant = self._router.resolve(persona_key, variant or self._current_variant, lang)

        voice_payload: dict[str, Any] = {
            **self._default_voice_params,
            **weights,
            **(voice_overrides or {}),
        }

        voice_field = voice_payload.get("voice") or voice_payload.get("voice_id") or "af_alloy"
        if isinstance(voice_field, str):
            voice_value: str | list[str] = voice_field
        elif isinstance(voice_field, Sequence):
            voice_value = [str(v) for v in voice_field]
        else:
            voice_value = "af_alloy"

        weights_field = voice_payload.get("weights")
        blend_weights: list[float] | None = None
        if isinstance(weights_field, Sequence):
            blend_weights = [float(w) for w in weights_field]
            if all(w == 0 for w in blend_weights):
                blend_weights = None
            elif abs(sum(blend_weights) - 1.0) > 1e-3:
                total = sum(blend_weights)
                if total:
                    blend_weights = [w / total for w in blend_weights]

        parameter_weights: dict[str, float] = {}
        recognised_keys = {"model", "voice", "voice_id", "response_format", "weights", "speed", "language"}

        payload: dict[str, object] = {
            "model": voice_payload.get("model", "kokoro"),
            "voice": voice_value,
            "input": text,
            "response_format": voice_payload.get("response_format", "wav"),
        }

        if blend_weights and isinstance(voice_value, list):
            payload["weights"] = blend_weights

        if "speed" in voice_payload:
            try:
                payload["speed"] = float(voice_payload["speed"])
            except (TypeError, ValueError):
                self._logger.warning("kokoro.tts.invalid_speed", speed=voice_payload["speed"])

        if "language" in voice_payload:
            payload["language"] = voice_payload["language"]
        elif lang:
            payload["language"] = lang

        for key, value in voice_payload.items():
            if key in recognised_keys:
                continue
            if isinstance(value, (int, float)):
                parameter_weights[key] = float(value)
            else:
                payload[key] = value

        if parameter_weights:
            existing_weights = payload.get("weights")
            if isinstance(existing_weights, dict):
                payload["weights"] = {**existing_weights, **parameter_weights}
            elif existing_weights is None:
                payload["weights"] = parameter_weights
            else:
                payload["weights_parameters"] = parameter_weights

        return payload, persona_key, resolved_variant, lang

    async def _fetch_audio(
        self,
        turn_id: str,
        payload: dict[str, object],
        persona_key: str,
        resolved_variant: str,
        lang: str,
    ) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        log_payload = {k: v for k, v in payload.items()}
        if isinstance(log_payload.get("input"), str) and len(log_payload["input"]) > 120:
            log_payload["input"] = log_payload["input"][:120] + "…"

        self._logger.info(
            "kokoro.tts.request",
            turn_id=turn_id,
            persona=persona_key,
            variant=resolved_variant,
            lang=lang,
            payload=log_payload,
        )
        try:
            async with self._client.stream("POST", self._base_url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                audio_chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    audio_chunks.append(chunk)
                return b"".join(audio_chunks)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 and payload.get("_sanitized") != True:
                self._logger.warning(
                    "kokoro.tts.fallback",
                    reason="bad_request",
                    status=exc.response.status_code,
                )
                fallback_payload: dict[str, object] = {
                    "model": payload.get("model", "kokoro"),
                    "voice": payload.get("voice", "af_alloy"),
                    "input": payload.get("input", ""),
                    "response_format": payload.get("response_format", "wav"),
                    "_sanitized": True,
                }
                if "language" in payload:
                    fallback_payload["language"] = payload["language"]
                if "speed" in payload:
                    fallback_payload["speed"] = payload["speed"]
                return await self._fetch_audio(turn_id, fallback_payload, persona_key, resolved_variant, lang)
            raise

    async def speak(
        self,
        turn_id: str,
        text: str,
        voice_overrides: Mapping[str, float | str] | None = None,
        persona: str | None = None,
        variant: str | None = None,
    ) -> float:
        payload, persona_key, resolved_variant, lang = self._build_request(text, voice_overrides, persona, variant)
        start = asyncio.get_running_loop().time()
        audio = await self._fetch_audio(turn_id, payload, persona_key, resolved_variant, lang)
        playback = await self._audio_output.play_bytes(audio, tag=f"tts:{turn_id}")
        end = asyncio.get_running_loop().time()
        if playback <= 0.0:
            playback = max(end - start, 0.0)
        return max(playback, 0.0)

    async def synthesize(
        self,
        text: str,
        persona: str | None = None,
        variant: str | None = None,
        voice_overrides: Mapping[str, float | str] | None = None,
    ) -> bytes:
        payload, persona_key, resolved_variant, lang = self._build_request(text, voice_overrides, persona, variant)
        return await self._fetch_audio("synthesize", payload, persona_key, resolved_variant, lang)

    async def _play_audio(self, audio: bytes) -> None:
        await self._audio_output.play_bytes(audio, tag="tts:legacy")

    async def aclose(self) -> None:
        await self._client.aclose()
