from __future__ import annotations

import asyncio
import json
import os
import time
from io import BytesIO
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs  # type: ignore
    HAS_LANGDETECT = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LANGDETECT = False
    class LangDetectException(Exception):
        pass
    def detect_langs(_: str):  # type: ignore
        raise LangDetectException()
from pydantic import BaseModel

from jarvis.audio.capture import AudioCapture
from jarvis.audio.output import AudioOutputController
from jarvis.audio.sfx import SoundEffectManager
from jarvis.audio.wakeword.base import WakeWordEngine
from jarvis.audio.wakeword.openwakeword import OpenWakeWordEngine
from jarvis.audio.wakeword.porcupine import PorcupineWakeWordEngine
from jarvis.config import AppSettings, load_settings, project_root
from jarvis.llm.providers.gemini import GeminiProvider
from jarvis.llm.providers.ollama import OllamaProvider
from jarvis.llm.router import Router
from jarvis.lang.script_detect import detect_lang_from_text
from jarvis.memory.store import MemoryStore, NullMemoryStore
from jarvis.orchestrator.events import WakeWordHit
from jarvis.persona import PersonaManager
from jarvis.orchestrator.state_machine import Orchestrator
from jarvis.orchestrator.maintenance import SelfMaintenanceScheduler
from jarvis.telemetry.logging import configure_logging, get_logger
from jarvis.telemetry.system_metrics import ResourceMonitor
from jarvis.telemetry.tracing import configure_tracing
from jarvis.transcription import VoskStream
from jarvis.transcription.base import TranscriptionEngine
from jarvis.tts.kokoro import KokoroTTSClient
from jarvis.tts.voice_router import VoiceRouter, load_router
from jarvis.ui.websocket import FloatingUIBridge


if HAS_LANGDETECT:
    DetectorFactory.seed = 0
INDIC_LANGUAGE_CODES: set[str] = {"hi", "mr", "pa", "ur", "bn", "ta", "te", "gu", "kn"}

VOICE_AGENT_PROMPT = """
You evaluate user utterances to decide if Jarvis should change its speaking voice.

Rules:
- Only change voice when the user explicitly requests a different tone, persona voice, gender, or refers to a predefined variant.
- Available voice personas and variants:
{available}
- The current utterance is: "{transcript}"

Respond with STRICT JSON (no prose, never wrap in code fences):
{{
  "change": true|false,
  "persona": "male"|"female"|null,
  "variant": "normal"|"alt2"|"special_weird"|null,
  "reason": "short sentence"
}}

If no change is required, set change=false and leave persona/variant null.
"""


settings = load_settings()
configure_logging(settings.telemetry.log_level)
configure_tracing("jarvis-voice-agent", settings.telemetry.otlp_endpoint)
logger = get_logger(__name__)

app = FastAPI(title="Jarvis Voice Agent")
ui_bridge = FloatingUIBridge()

origins = {settings.ui.floating_ui_origin}
if "localhost" in settings.ui.floating_ui_origin:
    origins.add(settings.ui.floating_ui_origin.replace("localhost", "127.0.0.1"))
app.include_router(ui_bridge.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

static_index = (project_root() / "Voice system" / "jarvis" / "ui" / "static" / "index.html").resolve()


@app.get("/ui", response_class=HTMLResponse)
async def index(_: Request) -> HTMLResponse:
    html = static_index.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.on_event("startup")
async def startup_event() -> None:
    app.state.runtime = await bootstrap_runtime()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    runtime = getattr(app.state, "runtime", None)
    if runtime:
        await runtime.shutdown()


@app.post("/manual/wake/start")
async def manual_wake_start() -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime:
        await runtime.manual_start()
    logger.info("manual.endpoint.start")
    return {"status": "ok"}


@app.post("/manual/wake/stop")
async def manual_wake_stop() -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime:
        await runtime.manual_stop()
    logger.info("manual.endpoint.stop")
    return {"status": "ok"}


@app.post("/asr/toggle-lang")
async def toggle_asr_language() -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime:
        new_engine = await runtime.toggle_asr_language()
        return {"status": "ok", "engine": new_engine}
    return {"status": "runtime-unavailable"}


class ChatRequest(BaseModel):
    text: str


class ASRSelectRequest(BaseModel):
    engine: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"text": "Assistant not ready yet."}
    reply = await runtime.handle_text(request.text)
    return reply


class VoiceDefaultRequest(BaseModel):
    persona: str
    variant: str


class VoiceSpeakRequest(BaseModel):
    persona: str
    variant: str | None = None
    text: str


class SoundEffectSelectRequest(BaseModel):
    event: str
    file: str


class SleepModeRequest(BaseModel):
    enabled: bool


class CameraEventRequest(BaseModel):
    event: str
    text: str | None = None


class LLMProviderRequest(BaseModel):
    provider: str


@app.get("/asr/options")
async def list_asr_options() -> dict[str, Any]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"engines": [], "current": "", "selected": "auto"}
    return {
        "engines": runtime.transcriber_options(),
        "current": runtime.current_transcriber(),
        "selected": runtime.transcriber_selection(),
    }


@app.post("/asr/select")
async def select_asr_engine(req: ASRSelectRequest) -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    try:
        engine = await runtime.set_transcriber(req.engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"engine": engine, "selected": runtime.transcriber_selection()}


@app.get("/persona/list")
async def persona_list() -> dict[str, object]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"personas": []}
    from jarvis.persona import PERSONAS
    return {"personas": [{"name": k, **v} for k, v in PERSONAS.items()]}


class PersonaSetRequest(BaseModel):
    name: str


@app.post("/persona/set")
async def persona_set(req: PersonaSetRequest) -> dict[str, object]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"status": "runtime-unavailable"}
    try:
        # Switch persona and speak a short confirmation
        await runtime._persona.activate(req.name, runtime._persona_language_target(runtime._language_hint(runtime._transcriber_name)))
        profile = runtime._persona.active_profile()
        await ui_bridge.publish_state("PERSONA", profile)
        # A brief audible confirmation
        await runtime._persona.tts.speak(
            "persona-switch",
            f"Persona set to {req.name}.",
            persona=str(profile.get("voice_persona", "male")),
            variant=profile.get("voice_variant") if profile.get("voice_variant") else None,
        )
        return {"status": "ok", "profile": profile}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/voice/personas")
async def voice_personas() -> dict[str, object]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"personas": []}
    return {"personas": runtime.voice_personas()}


@app.post("/voice/personas/set-default")
async def voice_set_default(req: VoiceDefaultRequest) -> dict[str, object]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    try:
        result = await runtime.set_voice_default(req.persona, req.variant)
        await ui_bridge.publish_state("PERSONA", runtime._persona.active_profile())
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/tts/speak")
async def tts_speak(req: VoiceSpeakRequest) -> StreamingResponse:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    audio = await runtime.preview_voice(req.persona, req.variant, req.text)
    if not audio:
        raise HTTPException(status_code=500, detail="tts returned empty audio")
    return StreamingResponse(
        BytesIO(audio),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=preview.wav"},
    )


@app.get("/admin/sfx")
async def list_sound_effects() -> dict[str, Any]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"events": [], "sleep": False}
    events = await runtime.sound_effects_snapshot()
    return {"events": events, "sleep": runtime.sleep_mode()}


@app.post("/admin/sfx/select")
async def select_sound_effect(req: SoundEffectSelectRequest) -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    try:
        await runtime.select_sound_effect(req.event, req.file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}


@app.post("/admin/sfx/upload")
async def upload_sound_effect(event: str = Form(...), file: UploadFile = File(...)) -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        saved = await runtime.add_sound_effect(event, file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"event": event, "file": saved}


@app.post("/admin/sleep")
async def set_sleep_mode(request: SleepModeRequest) -> dict[str, bool]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    state = await runtime.set_sleep_mode(request.enabled)
    return {"sleep": state}


@app.post("/api/events/camera")
async def camera_event(req: CameraEventRequest) -> dict[str, object]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"handled": False, "reason": "runtime-unavailable"}
    return await runtime.trigger_camera_event(req.event, req.text)


@app.get("/tts/sessions")
async def tts_sessions() -> dict[str, object]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"sessions": []}
    return {"sessions": runtime.tts_sessions()}


@app.get("/llm/provider")
async def get_llm_provider() -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    return {"provider": runtime.current_llm_provider()}


@app.post("/llm/provider")
async def set_llm_provider(req: LLMProviderRequest) -> dict[str, str]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime unavailable")
    try:
        provider = runtime.set_llm_provider(req.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"provider": provider}


@app.get("/llm/provider/options")
async def list_llm_providers() -> dict[str, list[str]]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"providers": []}
    return {"providers": runtime.llm_providers()}


class Runtime:
    def __init__(
        self,
        orchestrator: Orchestrator,
        router: Router,
        memory: MemoryStore,
        audio: AudioCapture,
        audio_output: AudioOutputController,
        soundfx: SoundEffectManager,
        transcriber: TranscriptionEngine,
        transcriber_name: str,
        transcriber_builder: Callable[[str | None], tuple[str, TranscriptionEngine]],
        transcriber_options: list[str],
        persona: PersonaManager,
        wakeword: WakeWordEngine | None,
        scheduler: SelfMaintenanceScheduler,
        monitor: ResourceMonitor,
        voice_router: VoiceRouter,
    ) -> None:
        self._orchestrator = orchestrator
        self._router = router
        self._memory = memory
        self._audio = audio
        self._audio_output = audio_output
        self._soundfx = soundfx
        self._transcriber: TranscriptionEngine | None = transcriber
        self._transcriber_name = transcriber_name
        self._transcriber_builder = transcriber_builder
        self._transcriber_options = sorted(transcriber_options)
        self._wakeword = wakeword
        self._persona = persona
        self._scheduler = scheduler
        self._monitor = monitor
        self._voice_router = voice_router
        self._tasks: set[asyncio.Task] = set()
        self._transcription_task: asyncio.Task | None = None
        self._wake_hit: WakeWordHit | None = None
        self._manual_active = False
        self._manual_active_deadline = 0.0
        self._listening_active = False
        self._received_audio_since_activation = False
        self._silence_duration = 0.0
        self._silence_timeout = 1.2  # seconds of silence to auto-complete
        self._min_listen_guard = 2.4
        self._listening_guard_until = 0.0
        self._transcriber_lock: str | None = None
        self._sleeping = False
        self._logger = get_logger(__name__)
        self._shutdown_in_progress = False
        self._language_scores: dict[str, float] = {}
        self._last_language: str | None = None
        self._switch_lock = asyncio.Lock()
        self._tts_sessions: list[dict[str, object]] = []
        self._agent_step_id = 0

    async def start(self) -> None:
        self._logger.info("runtime.starting", engine=self._transcriber_name)
        audio_task = asyncio.create_task(self._run_audio_loop(), name="audio-loop")
        self._tasks.add(audio_task)
        if self._transcriber:
            self._transcription_task = asyncio.create_task(
                self._transcription_loop(self._transcriber, self._transcriber_name),
                name=f"stt-loop:{self._transcriber_name}",
            )
            self._tasks.add(self._transcription_task)
        if self._wakeword:
            wake_task = asyncio.create_task(self._run_wakeword_loop(), name="wakeword-loop")
            self._tasks.add(wake_task)
        await ui_bridge.publish_state("PERSONA", self._persona.active_profile())
        await self._scheduler.start()
        await self._monitor.start()
        await self._soundfx.play("boot_sequence", await_completion=False)

    async def shutdown(self) -> None:
        self._logger.info("runtime.shutdown.start")
        await self._audio.stop()
        if self._transcriber:
            await self._transcriber.close()
        if self._wakeword:
            await self._wakeword.close()
        await self._monitor.shutdown()
        await self._scheduler.shutdown()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._transcription_task = None
        self._transcriber = None
        self._logger.info("runtime.shutdown.complete")

    def voice_personas(self) -> dict[str, dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for persona in self._voice_router.personas():
            summary[persona] = {
                "variants": self._voice_router.variants_for(persona),
                "default_variant": self._voice_router.default_variant(persona),
            }
        return summary

    def _refresh_manual_flag(self) -> None:
        if not self._manual_active:
            return
        if self._manual_active_deadline and time.time() > self._manual_active_deadline:
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._logger.info("manual.guard.expired")

    def _begin_listening(self, keyword: str) -> None:
        now = time.time()
        self._listening_active = True
        self._silence_duration = 0.0
        self._received_audio_since_activation = False
        self._listening_guard_until = now + self._min_listen_guard
        self._logger.debug(
            "listen.guard.start",
            keyword=keyword,
            guard_seconds=self._min_listen_guard,
        )

    def _stop_listening(self) -> None:
        self._listening_active = False
        self._silence_duration = 0.0
        self._received_audio_since_activation = False
        self._listening_guard_until = 0.0

    async def set_voice_default(self, persona: str, variant: str) -> dict[str, object]:
        self._voice_router.set_default_variant(persona, variant)
        profile = self._persona.active_profile()
        if profile.get("voice_persona") == persona and profile.get("voice_variant") is None:
            await self._persona.tts.configure_voice(persona, None)
        return {
            "persona": persona,
            "default_variant": self._voice_router.default_variant(persona),
        }

    async def preview_voice(self, persona: str, variant: str | None, text: str) -> bytes:
        return await self._persona.tts.synthesize(
            text,
            persona=persona,
            variant=variant,
        )

    async def trigger_camera_event(self, event: str, text_override: str | None = None) -> dict[str, object]:
        cfg = self._voice_router.camera_event(event)
        if not cfg:
            return {"handled": False, "reason": "unknown_event"}
        persona = cfg.get("persona")
        variant = cfg.get("variant")
        line = text_override if text_override is not None else cfg.get("line", "")
        if persona:
            await self._persona.tts.speak(
                f"camera-{event}",
                str(line),
                persona=str(persona),
                variant=str(variant) if variant else None,
            )
            return {"handled": True, "persona": persona, "variant": variant, "line": line}
        return {"handled": False, "reason": "invalid_config"}

    def current_llm_provider(self) -> str:
        return self._router.current_provider()

    def set_llm_provider(self, provider: str) -> str:
        return self._router.set_default(provider)

    def llm_providers(self) -> list[str]:
        return self._router.available()

    def tts_sessions(self) -> list[dict[str, object]]:
        return list(self._tts_sessions)

    async def _publish_llm_step(
        self,
        name: str,
        status: str,
        provider: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        self._agent_step_id += 1
        payload = {
            "id": self._agent_step_id,
            "name": name,
            "status": status,
            "provider": provider,
        }
        if detail:
            payload.update(detail)
        await ui_bridge.publish_state("LLM_AGENT", payload)

    def _record_tts_session(self, mode: str, duration: float, transcript: str) -> None:
        if duration <= 0:
            return
        self._tts_sessions.append(
            {
                "timestamp": time.time(),
                "mode": mode,
                "duration": duration,
                "transcript": transcript,
            }
        )

    async def _handle_voice_change_agent(self, transcript: str) -> bool:
        available = {
            persona: self._voice_router.variants_for(persona)
            for persona in self._voice_router.personas()
        }
        prompt = VOICE_AGENT_PROMPT.format(
            available=json.dumps(available, indent=2),
            transcript=transcript,
        )
        await self._publish_llm_step("voice_agent", "queued", detail={"utterance": transcript})
        try:
            plan = await self._router.decide(prompt, {})
            provider_name = plan.provider.name
            await self._publish_llm_step("voice_agent", "running", provider=provider_name)
            resp = await plan.provider.chat(prompt)
            await self._publish_llm_step("voice_agent", "complete", provider=provider_name)
        except Exception as exc:
            self._logger.warning("voice.agent.error", error=str(exc))
            await self._publish_llm_step("voice_agent", "error", detail={"error": str(exc)})
            return False

        text = resp.text.strip()
        if text.startswith("```"):
            text = text.strip("`\n ")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self._logger.warning("voice.agent.parse_error", payload=text, error=str(exc))
            await self._publish_llm_step("voice_agent", "error", detail={"error": "invalid_json"})
            return False

        if not isinstance(data, dict) or not data.get("change"):
            return False

        persona = data.get("persona") or data.get("voice_persona")
        variant = data.get("variant")
        if persona not in available:
            self._logger.warning("voice.agent.invalid_persona", persona=persona)
            return False
        if variant is not None and variant not in available[persona]:
            self._logger.warning("voice.agent.invalid_variant", persona=persona, variant=variant)
            return False

        if variant is None:
            variant = self._voice_router.default_variant(persona)

        try:
            self._voice_router.set_default_variant(persona, variant)
        except ValueError as exc:
            self._logger.warning("voice.agent.apply_failed", error=str(exc))
            return False

        await self._persona.override_voice(persona, variant)
        await ui_bridge.publish_state("PERSONA", self._persona.active_profile())
        await self._publish_llm_step(
            "voice_agent",
            "applied",
            provider=None,
            detail={"persona": persona, "variant": variant, "reason": data.get("reason")},
        )
        return True

    async def _run_audio_loop(self) -> None:
        async for frame in self._audio.frames():
            now_ts = frame.ts or time.time()
            if frame.energy > self._audio.energy_threshold:
                self._logger.debug("audio.energy.detected", energy=frame.energy)

            # Only show LISTENING when actively listening; otherwise surface IDLE with meter
            if self._listening_active:
                await ui_bridge.publish_state(
                    "LISTENING",
                    {
                        "vad": frame.vad,
                        "active": True,
                        "persona": self._persona.active_name,
                        "engine": self._transcriber_name,
                        "asr_engine": self._transcriber_name,
                        "language": self._language_hint(self._transcriber_name),
                        "audio_level": frame.energy,
                    },
                )
            else:
                await ui_bridge.publish_state(
                    "IDLE",
                    {
                        "vad": False,
                        "active": False,
                        "persona": self._persona.active_name,
                        "engine": self._transcriber_name,
                        "asr_engine": self._transcriber_name,
                        "language": self._language_hint(self._transcriber_name),
                        "audio_level": frame.energy,
                    },
                )

            if not self._listening_active:
                continue

            transcriber = self._transcriber
            if not transcriber:
                await self._ensure_transcriber_ready()
                transcriber = self._transcriber
                if not transcriber:
                    continue

            await transcriber.enqueue_audio(frame.pcm16le, frame.ts, frame.vad)

            if frame.vad:
                self._received_audio_since_activation = True
                self._silence_duration = 0.0
            else:
                self._silence_duration += self._audio.frame_ms / 1000
                guard_active = (
                    not self._received_audio_since_activation or now_ts < self._listening_guard_until
                )
                if guard_active:
                    continue
                if self._silence_duration >= self._silence_timeout:
                    self._logger.info("listen.silence_timeout", duration=self._silence_duration)
                    self._stop_listening()
                    await transcriber.enqueue_audio(b"", time.time(), False, force=True)

    async def _transcription_loop(self, transcriber: TranscriptionEngine, engine_name: str) -> None:
        current_text = ""
        try:
            async for chunk in transcriber.stream():
                if self._transcriber is not transcriber:
                    break

                text = chunk.text.strip()
                if text:
                    current_text = text

                payload = {
                    "transcript": current_text or text,
                    "range_ms": chunk.range_ms,
                    "engine": engine_name,
                    "persona": self._persona.active_name,
                    "persona_profile": self._persona.active_profile(),
                    "is_final": chunk.is_final,
                    "asr_engine": engine_name,
                    "language": self._language_hint(engine_name),
                    "audio_level": 0.0,
                }
                await ui_bridge.publish_state("LISTENING", payload)

                if not chunk.is_final:
                    continue

                transcript = text or current_text
                current_text = ""
                await self._handle_final_transcript(transcript, engine_name)
        except Exception as exc:
            self._logger.error("transcription.loop.error", engine=engine_name, error=str(exc))
            await self._soundfx.play("small_error", await_completion=False)
            if self._transcriber is transcriber and not engine_name.startswith("vosk"):
                await self._swap_transcriber("vosk-en-in")
        finally:
            if self._transcriber is transcriber:
                self._logger.info("transcription.loop.exit", engine=engine_name)
                self._transcriber = None

    async def _run_wakeword_loop(self) -> None:
        assert self._wakeword is not None
        async for hit in self._wakeword.run():
            self._refresh_manual_flag()
            if self._manual_active:
                self._logger.info("wakeword.ignored", keyword=hit.keyword, reason="manual_active")
                continue
            if self._sleeping:
                self._sleeping = False
                await self._soundfx.play("boot_sequence", await_completion=False)
            current_tag = self._audio_output.current_tag()
            if current_tag and current_tag.startswith("tts:"):
                await self._audio_output.stop(current_tag)
                await self._soundfx.play("interruption", await_completion=False)
            await self._soundfx.play("wakeword", await_completion=False)
            self._wake_hit = hit
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._begin_listening(hit.keyword)
            await self._ensure_transcriber_ready()
            self._logger.info("wakeword.detected", keyword=hit.keyword, confidence=hit.confidence)
            await ui_bridge.publish_state("HOTWORD_HEARD", {"keyword": hit.keyword, "confidence": hit.confidence})
            await ui_bridge.publish_state(
                "LISTENING",
                {
                    "vad": False,
                    "active": True,
                    "keyword": hit.keyword,
                    "engine": self._transcriber_name,
                    "persona": self._persona.active_name,
                    "asr_engine": self._transcriber_name,
                    "language": self._language_hint(self._transcriber_name),
                    "audio_level": 0.0,
                },
            )

    async def manual_start(self) -> None:
        if self._manual_active:
            self._logger.info("manual.start.ignored", reason="already_active")
            return
        self._wake_hit = WakeWordHit(ts=time.time(), keyword="manual", confidence=1.0, buffer_ref_ms=4000)
        self._manual_active = True
        self._manual_active_deadline = time.time() + 15.0
        self._sleeping = False
        self._begin_listening("manual")
        self._logger.info("manual.start")
        await self._ensure_transcriber_ready()
        await ui_bridge.publish_state("HOTWORD_HEARD", {"keyword": "manual", "confidence": 1.0})
        await ui_bridge.publish_state(
            "LISTENING",
            {
                "vad": False,
                "active": True,
                "keyword": "manual",
                "engine": self._transcriber_name,
                "persona": self._persona.active_name,
                "asr_engine": self._transcriber_name,
                "language": self._language_hint(self._transcriber_name),
                "audio_level": 0.0,
            },
        )

    async def manual_stop(self) -> None:
        if not self._manual_active and not self._listening_active:
            self._logger.info("manual.stop.ignored", reason="inactive")
            return
        transcriber = self._transcriber
        if transcriber:
            await transcriber.enqueue_audio(b"", time.time(), False, force=True)
        self._manual_active = False
        self._manual_active_deadline = 0.0
        self._stop_listening()
        self._wake_hit = None
        self._logger.info("manual.stop")
        await ui_bridge.publish_state("IDLE", {"keyword": "manual", "audio_level": 0.0})

    async def toggle_asr_language(self) -> str:
        current = self._transcriber_name or ""
        target = "vosk-hi" if not current.startswith("vosk-hi") else "vosk-en-in"
        self._transcriber_lock = target
        await self._swap_transcriber(target)
        await self._persona.set_language(
            self._persona_language_target(self._language_hint(self._transcriber_name))
        )
        return self._transcriber_name or target

    def transcriber_options(self) -> list[str]:
        return ["auto"] + self._transcriber_options

    def current_transcriber(self) -> str:
        return self._transcriber_lock or (self._transcriber_name or "")

    def transcriber_selection(self) -> str:
        return self._transcriber_lock or "auto"

    async def set_transcriber(self, engine: str) -> str:
        if engine == "auto":
            self._transcriber_lock = None
            self._logger.info("transcription.engine.lock.cleared")
            return self._transcriber_name or ""
        if engine not in self._transcriber_options:
            raise ValueError(f"Unknown transcription engine '{engine}'")
        self._transcriber_lock = engine
        await self._swap_transcriber(engine)
        return self._transcriber_name

    async def sound_effects_snapshot(self) -> list[dict[str, Any]]:
        return await self._soundfx.snapshot()

    async def select_sound_effect(self, event: str, filename: str) -> None:
        await self._soundfx.select(event, filename)

    async def add_sound_effect(self, event: str, original_name: str, data: bytes) -> str:
        return await self._soundfx.add(event, original_name, data)

    async def set_sleep_mode(self, enabled: bool) -> bool:
        if enabled == self._sleeping:
            return self._sleeping
        self._sleeping = enabled
        if enabled:
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._stop_listening()
            await self._soundfx.play("sleep_mode", await_completion=False)
        return self._sleeping

    def sleep_mode(self) -> bool:
        return self._sleeping

    async def _handle_final_transcript(self, transcript: str, engine_name: str) -> None:
        transcript = (transcript or "").strip()
        if not transcript:
            self._logger.info("transcription.empty_final", engine=engine_name)
            self._wake_hit = None
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._stop_listening()
            self._last_language = None
            return

        if await self._handle_persona_command(transcript):
            self._wake_hit = None
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._stop_listening()
            return

        if await self._handle_voice_change_agent(transcript):
            self._wake_hit = None
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._stop_listening()
            return

        wake_hit = self._wake_hit
        if wake_hit is None and self._wakeword is None:
            wake_hit = WakeWordHit(ts=time.time(), keyword="auto", confidence=1.0, buffer_ref_ms=4000)

        if not self._shutdown_in_progress and "hey creo shutdown" in transcript.lower():
            self._logger.info("shutdown.command.detected", transcript=transcript)
            await ui_bridge.publish_state("MAINTENANCE", {"prompt": "Shutting down..."})
            self._shutdown_in_progress = True
            self._manual_active = False
            self._manual_active_deadline = 0.0
            self._wake_hit = None
            self._stop_listening()
            asyncio.create_task(self._perform_exit())
            return

        if wake_hit:
            self._logger.info(
                "runtime.turn.process",
                transcript=transcript,
                keyword=wake_hit.keyword,
                engine=engine_name,
            )
            audio_blob = self._audio.last_audio(6.0)
            audio_ref = f"ringbuffer://last6s?bytes={len(audio_blob)}"
            duration = await self._orchestrator.handle_turn(
                transcript=transcript,
                wake_hit=wake_hit,
                audio_refs=[audio_ref],
                context_factory=lambda: self._memory.recall(None),
            )
            self._record_tts_session("voice", duration, transcript)
        else:
            self._logger.info("runtime.turn.process.text_only", transcript=transcript, engine=engine_name)
            message = await self._orchestrator.run_text(transcript, lambda: self._memory.recall(None))
            self._record_tts_session("text", message.tts_duration or 0.0, transcript)

        self._wake_hit = None
        self._manual_active = False
        self._manual_active_deadline = 0.0
        self._stop_listening()
        await self._maybe_switch_engine(transcript, engine_name)
        lang_hint = self._last_language or self._language_hint(self._transcriber_name)
        if lang_hint == "--":
            lang_hint = None
        self._last_language = lang_hint
        await self._persona.set_language(self._persona_language_target(lang_hint))

    async def _maybe_switch_engine(self, transcript: str, engine_name: str) -> None:
        if self._transcriber_lock:
            return
        target = self._target_engine_for_transcript(transcript, engine_name)
        if not target:
            return
        self._logger.info(
            "transcription.language.switch",
            current=engine_name,
            target=target,
            scores=self._language_scores,
        )
        await self._swap_transcriber(target)

    def _target_engine_for_transcript(self, transcript: str, engine_name: str) -> str | None:
        if len(transcript) < 4:
            return None
        script_lang = detect_lang_from_text(transcript)
        if script_lang == "hi" and engine_name != "vosk-hi":
            return "vosk-hi"
        if script_lang == "en" and engine_name != "vosk-en-in" and engine_name != "vosk-en":
            return "vosk-en-in"
        scores = self._detect_languages(transcript)
        indic_score = max((scores.get(code, 0.0) for code in INDIC_LANGUAGE_CODES), default=0.0)
        english_score = max((scores.get(code, 0.0) for code in ("en", "en-in", "en-us")), default=0.0)

        if engine_name not in {"vosk-hi"} and indic_score >= 0.25 and indic_score > english_score:
            return "vosk-hi"
        if not engine_name.startswith("vosk-en") and english_score >= 0.6 and english_score > indic_score + 0.1:
            return "vosk-en-in"
        return None

    def _language_hint(self, engine_name: str | None) -> str:
        if self._last_language:
            return self._last_language
        if not engine_name:
            return "--"
        if engine_name.startswith("vosk-hi"):
            return "hi"
        if engine_name.startswith("vosk-en"):
            return "en-IN"
        return "en"

    def _persona_language_target(self, lang_hint: str | None) -> str:
        if not lang_hint:
            return "english"
        normalized = lang_hint.lower()
        if normalized in {"en", "en-in", "en-us", "en-gb"}:
            return "english"
        if any(normalized.startswith(code) for code in INDIC_LANGUAGE_CODES) or normalized.startswith("hi"):
            return "hinglish"
        return "english"

    def _detect_languages(self, transcript: str) -> dict[str, float]:
        try:
            detections = detect_langs(transcript)
        except Exception:
            self._language_scores = {}
            self._last_language = None
            return {}
        scores = {det.lang: det.prob for det in detections}
        self._language_scores = scores
        if scores:
            self._last_language = max(scores, key=scores.get)
        else:
            self._last_language = None
        return scores

    async def _swap_transcriber(self, target_engine: str) -> None:
        async with self._switch_lock:
            if target_engine == self._transcriber_name:
                return

            self._logger.info(
                "transcription.engine.switch.start",
                current=self._transcriber_name,
                target=target_engine,
            )

            try:
                new_name, new_transcriber = self._transcriber_builder(target_engine)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error("transcription.engine.switch.failed", target=target_engine, error=str(exc))
                await self._soundfx.play("small_error", await_completion=False)
                return

            old_transcriber = self._transcriber
            old_task = self._transcription_task

            self._transcriber = new_transcriber
            self._transcriber_name = new_name
            self._last_language = None

            new_task = asyncio.create_task(
                self._transcription_loop(new_transcriber, new_name),
                name=f"stt-loop:{new_name}",
            )
            self._transcription_task = new_task
            self._tasks.add(new_task)

            if old_transcriber:
                await old_transcriber.close()

            if old_task:
                await asyncio.gather(old_task, return_exceptions=True)
                self._tasks.discard(old_task)

            self._logger.info("transcription.engine.switch.complete", engine=new_name)
            await ui_bridge.publish_state(
                "LISTENING",
                {
                    "vad": False,
                    "active": self._listening_active,
                    "engine": self._transcriber_name,
                    "persona": self._persona.active_name,
                    "asr_engine": self._transcriber_name,
                    "language": self._language_hint(self._transcriber_name),
                    "audio_level": 0.0,
                },
            )
            await self._persona.set_language(
                self._persona_language_target(self._language_hint(self._transcriber_name))
            )

    async def _ensure_transcriber_ready(self) -> None:
        if self._transcriber:
            return
        async with self._switch_lock:
            if self._transcriber:
                return
            try:
                name, instance = self._transcriber_builder(self._transcriber_name)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error("transcription.ensure.failed", error=str(exc))
                return
            self._transcriber = instance
            self._transcriber_name = name
            task = asyncio.create_task(
                self._transcription_loop(instance, name),
                name=f"stt-loop:{name}",
            )
            self._transcription_task = task
            self._tasks.add(task)

    async def _handle_persona_command(self, transcript: str) -> bool:
        target = self._persona.match_command(transcript)
        if not target:
            return False

        if target == self._persona.active_name:
            self._logger.info("persona.command.noop", persona=target)
            await ui_bridge.publish_state("PERSONA", self._persona.active_profile())
            return True

        try:
            language = self._persona_language_target(
                self._last_language or self._language_hint(self._transcriber_name)
            )
            await self._persona.activate(target, language)
        except ValueError as exc:
            self._logger.warning("persona.command.invalid", transcript=transcript, error=str(exc))
            return False

        await ui_bridge.publish_state("PERSONA", self._persona.active_profile())
        self._logger.info("persona.command.fulfilled", persona=target)
        return True

    async def _perform_exit(self) -> None:
        await self._soundfx.play("exit", await_completion=True)
        await asyncio.sleep(0.2)
        try:
            await self.shutdown()
        finally:
            os._exit(0)

    async def handle_text(self, text: str) -> dict[str, str]:
        self._logger.info("runtime.text_turn.start")
        message = await self._orchestrator.run_text(text, lambda: self._memory.recall(None))
        return {"text": message.text}


def collect_transcriber_models(settings: AppSettings) -> dict[str, str]:
    models: dict[str, str] = {}
    if settings.transcription.vosk_model_path_en_in:
        models["vosk-en-in"] = settings.transcription.vosk_model_path_en_in
    if settings.transcription.vosk_model_path:
        models.setdefault("vosk-en-in", settings.transcription.vosk_model_path)
    if settings.transcription.vosk_model_path_hi:
        models["vosk-hi"] = settings.transcription.vosk_model_path_hi
    return models


def build_transcriber(
    settings: AppSettings,
    preferred: str | None = None,
    available: dict[str, str] | None = None,
) -> tuple[str, TranscriptionEngine]:
    vosk_models = dict(available or collect_transcriber_models(settings))

    def normalize(engine: str | None) -> str:
        # Force Vosk-only usage; map empty/cheetah to Vosk default if available
        if not engine or engine == "cheetah":
            if "vosk-en-in" in vosk_models:
                return "vosk-en-in"
            if vosk_models:
                return next(iter(vosk_models))
            return "vosk"
        if engine.startswith("vosk"):
            if engine in vosk_models:
                return engine
            # Fall back to default Vosk variant if explicit one missing
            if "vosk-en-in" in vosk_models:
                return "vosk-en-in"
            if vosk_models:
                return next(iter(vosk_models))
            return "vosk"
        return engine

    order: list[str] = []
    if preferred:
        order.append(normalize(preferred))
    default_engine = normalize(settings.transcription.engine)
    if default_engine not in order:
        order.append(default_engine)
    for fallback in ("vosk-en-in", "vosk-hi"):
        normalized = normalize(fallback)
        if normalized not in order:
            order.append(normalized)

    last_exc: Exception | None = None
    for engine_name in order:
        try:
            if engine_name.startswith("vosk"):
                if VoskStream is None:  # pragma: no cover - optional dependency
                    raise ModuleNotFoundError("vosk is not installed")
                model_path = vosk_models.get(engine_name)
                if not model_path:
                    raise ValueError(f"Model path for {engine_name} not configured")
                transcriber = VoskStream(model_path=model_path, sample_rate=settings.realtime_stt.sample_rate)
            else:
                continue

            logger.info("transcription.engine.selected", engine=engine_name)
            return engine_name, transcriber
        except Exception as exc:
            logger.error("transcription.engine.failed", engine=engine_name, error=str(exc))
            last_exc = exc

    raise RuntimeError("No transcription engine available") from last_exc


async def bootstrap_runtime() -> Runtime:
    audio_output = AudioOutputController()
    sound_effects = SoundEffectManager(
        base_dir=project_root() / "Voice system" / "sound effects",
        audio_output=audio_output,
        config_path=project_root() / "Voice system" / "config" / "sound_effects.yml",
    )
    await sound_effects.initialize()
    try:
        memory: MemoryStore | NullMemoryStore = MemoryStore(settings.database.dsn, settings.database.max_pool_size)
        await memory.init()
    except Exception as exc:
        logger.error("memory.init.failed", error=str(exc))
        await sound_effects.play("big_error", await_completion=False)
        memory = NullMemoryStore()
        await memory.init()

    ollama = OllamaProvider(settings.llm.ollama_host)
    gemini = GeminiProvider(settings.llm.gemini_api_key) if settings.llm.gemini_api_key else None
    router = Router(ollama=ollama, gemini=gemini)

    voice_router = load_router()
    kokoro = KokoroTTSClient(
        settings.kokoro.base_url,
        settings.kokoro.api_key,
        {},
        router=voice_router,
        audio_output=audio_output,
    )

    persona_manager = PersonaManager(kokoro)
    await persona_manager.initialize()

    orchestrator = Orchestrator(
        memory=memory,
        router=router,
        tts=kokoro,
        persona=persona_manager,
        tool_executor=ToolExecutorStub(),
        ui_bridge=ui_bridge,
    )

    audio = AudioCapture(
        samplerate=settings.realtime_stt.sample_rate,
        frame_ms=settings.realtime_stt.frame_ms,
        energy_threshold=settings.realtime_stt.energy_threshold,
        device=settings.realtime_stt.input_device,
    )
    audio.start()

    vosk_models = collect_transcriber_models(settings)
    engine_name, transcriber = build_transcriber(settings, available=vosk_models)

    def transcriber_builder(preferred: str | None = None) -> tuple[str, TranscriptionEngine]:
        return build_transcriber(settings, preferred, vosk_models)

    wakeword_engine: WakeWordEngine | None = None
    if settings.wakeword.engine == "porcupine" and settings.wakeword.porcupine_keyword_path:
        wakeword_engine = PorcupineWakeWordEngine(
            keyword_path=settings.wakeword.porcupine_keyword_path,
            model_path=settings.wakeword.porcupine_model_path,
            access_key=settings.wakeword.porcupine_access_key,
        )
    elif settings.wakeword.engine == "openwakeword":
        wakeword_engine = OpenWakeWordEngine(keyword=settings.wakeword.keyword)

    scheduler = SelfMaintenanceScheduler(ui_bridge=ui_bridge)
    monitor = ResourceMonitor(interval_seconds=settings.telemetry.resource_sample_seconds, ui_bridge=ui_bridge)

    runtime = Runtime(
        orchestrator,
        router,
        memory,
        audio,
        audio_output,
        sound_effects,
        transcriber,
        engine_name,
        transcriber_builder,
        list(vosk_models.keys()),
        persona_manager,
        wakeword_engine,
        scheduler,
        monitor,
        voice_router,
    )
    await runtime.start()
    logger.info("runtime.started")
    return runtime


class ToolExecutorStub:
    async def invoke(self, call: dict) -> dict:
        logger.info("tool.stub.invoke", call=call)
        return {"tool": call.get("tool"), "result": "stub"}
