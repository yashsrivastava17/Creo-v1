# V2 Codex Handoff

Welcome to the second major iteration of the Creo voice agent. This document captures the high-level system update and links out to detailed notes living in this folder.

## What Changed in V2

- **Modular speech-to-text pipeline**: Picovoice Cheetah handles low-latency English captions, while Vosk provides resilient Hinglish fallback. Engines are hot-swappable at runtime, with automatic detection and handover based on language mix.
- **Persona-aware runtime**: A new persona manager keeps Kokoro TTS, the LLM providers, and the UI aligned. Voice switches can be requested verbally (e.g., “switch to fun voice”) and propagate instantly through the stack.
- **Sharper listening loop**: Silence detection is trimmed to 0.6 s, manual stop flushes transcription buffers, and fallback creation is thread-safe so the audio loop never blocks.
- **UI telemetry upgrades**: The floating ring now displays friendly state labels, live transcripts land in the sidebar with persona/voice metadata, and the `/` shortcut works globally thanks to a focusable body.
- **Lean dependencies**: Faster-Whisper is gone. New deps are `pvcheetah`, `vosk`, `langdetect`, and upgraded `soundfile` for smoother audio interop.

## Quick Start Checklist

1. Install dependencies: `pip install -e .`
2. Provide the Picovoice access key and (optional) model path via `.env` (see `environment_and_env.md`).
3. Download the Vosk model referenced by `VOSK_MODEL_PATH` (default points to `Voice system/models/vosk-model-small-en-in`).
4. Restart the service: `python -m uvicorn jarvis.main:app --port 8010`
5. Visit `/ui`, use the wake word or click the ring, and verify that persona + transcription data populate the sidebar.

## Where to Look Next

- [`architecture.md`](architecture.md) – End-to-end data flow from microphone to Kokoro playback.
- [`resource_management.md`](resource_management.md) – Memory and CPU envelopes, plus tunables to stay inside your M3 Pro comfort zone.
- [`models.md`](models.md) – All engines in play: ASR, TTS, and LLM providers with tuning notes.
- [`orchestrator_capabilities.md`](orchestrator_capabilities.md) – State-machine map and how personas converge with routing decisions.
- [`memory_models.md`](memory_models.md) – Persistence strategy, fallbacks, and guardrails.
- [`audio_pipeline.md`](audio_pipeline.md) – Buffering, VAD, silence thresholds, and engine switching behaviour.
- [`environment_and_env.md`](environment_and_env.md) – Virtualenv tips plus every environment flag documented.

## Operational Notes

- Keep `TRANSCRIPTION_ENGINE` in `.env` to your preferred primary (`cheetah` or `vosk`); fallback happens automatically.
- Persona swaps punch through immediately, but you can always confirm the current selection via the Persona card in the floating UI.
- When testing failover, disconnect the network or corrupt the Cheetah key on purpose—the runtime will log the failure and drop into Vosk without user intervention.
- Before shipping to another machine, run `pip install pvcheetah vosk langdetect soundfile` explicitly; those wheels can be large and should be cached ahead of time.

Stay curious, and ping the linked docs for subsystem deep dives.
