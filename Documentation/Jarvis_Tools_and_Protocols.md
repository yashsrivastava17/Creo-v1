# Jarvis Voice Assistant — Tools & Protocols

This note tracks the modules and operational routines that sit on top of the V1 orchestrator scaffold.

## Transcription

- **Primary (local)**: Picovoice Cheetah via `jarvis.transcription.cheetah.CheetahStream`. Provide `CHEETAH_ACCESS_KEY` (and optionally `CHEETAH_MODEL_PATH`) in `.env`.
- **Automatic fallback**: Vosk (`jarvis.transcription.vosk.VoskStream`) kicks in for Hinglish-heavy speech or whenever Cheetah fails to initialise.
- **Optional (remote)**: [`RealtimeSTT`](https://github.com/rhasspy/realtimestt) remains available if you prefer an external WebSocket ASR backend, but it is no longer the default path.

## Wake Word

- **Porcupine** (default keyword: `jarvis` for now; switch back to `creo` after the new model is tuned). Requires Picovoice access key + `.ppn` keyword file paths in `.env`.
- **openWakeWord** fallback by setting `WAKEWORD_ENGINE=openwakeword` if you want an OSS detector.

## TTS

- **Kokoro** HTTP endpoint at `http://localhost:8880/v1/audio/speech`. Voice presets from the handoff doc live under `jarvis/tts/voices.py` and can be updated at runtime.
- Responses are analysed for script at runtime: any Devanagari character switches the Kokoro router to Hindi weights for the current persona & variant; otherwise English weights are used.
- The LLM agent can also suggest persona/variant changes conversationally. Valid commands result in a JSON payload which is executed against the voice router.
- The `/voice/personas` REST endpoints (and the console voice panel) expose preview + default management and persist runtime choices.

## Memory & Persistence

- **Database**: Postgres + `pgvector` (see `.env.example`). The current store persists profile facts, ephemerals, turns, and tool outputs.
- Embedding pipeline hooks are stubbed in `jarvis/memory/store.py`; wire your favourite encoder before promoting facts.

## Self-Maintenance & RLHF Hooks

- `jarvis.orchestrator.maintenance.SelfMaintenanceScheduler` emits a `MAINTENANCE` state every four hours. The UI surfaces a prompt asking the user for feedback or preference updates. Extend this to capture explicit RLHF or evaluation data.
- Use the scheduler to call into bespoke routines (e.g., nightly log review, dataset curation, or model fine-tuning triggers).

## Resource Monitoring

- `jarvis.telemetry.system_metrics.ResourceMonitor` samples CPU/RAM usage (via `psutil`) at the interval defined by `RESOURCE_SAMPLE_SECONDS`. Samples are logged and pushed to the UI as `RESOURCE` events so you can spot spikes when Kokoro, Ollama, Cheetah, or Vosk compete for headroom.

## Manual Push-to-talk & Shutdown

- The floating UI ring posts to `/manual/wake/start` and `/manual/wake/stop`, allowing touch-driven activation in addition to wake words. Manual sessions time out automatically after ~1 second of silence.
- Saying “hey creo shutdown” invokes the manual stop path and exits the service cleanly; useful when running hands-free.

## Tooling Roadmap

- **LLM Router** (`jarvis.llm.router.Router`): chooses between Ollama (default) and Gemini; extend with additional providers or tool plans.
- **Tool Executor Stub**: replace the placeholder in `jarvis.main.ToolExecutorStub` with concrete integrations such as web search, calendar, home automation, or RAG notebooks.
- **Telemetry**: JSON logs (Structlog) + optional OTLP traces. Wire dashboards/alerting depending on deployment target.

## Next Actions

1. Implement active-learning storage for RLHF signals captured during `MAINTENANCE` prompts.
2. Add guardrails for long-term memory writes before promoting facts.
3. Evaluate real-time VAD options (Silero / WebRTC) to improve chunking for both Cheetah and Vosk.
