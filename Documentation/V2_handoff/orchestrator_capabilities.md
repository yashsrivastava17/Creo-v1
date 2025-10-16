# Orchestrator Capabilities

The orchestrator coordinates memory, LLM routing, persona-aware speech, and UI signalling. This map summarises how a turn travels through the state machine.

## Turn Lifecycle

1. **Wake / Manual Trigger**
   - Wake word loop emits `HOTWORD_HEARD`, and the runtime primes `_wake_hit`.
   - Manual start forces an artificial wake hit with confidence 1.0.

2. **Listening**
   - Audio frames enter the active transcriber queue.
   - Partial transcripts stream to the UI with persona + engine metadata.
   - Silence timeout or manual stop forces a flush to produce a final transcript.

3. **Persona Command Intercept (optional)**
   - Before contacting an LLM, `_handle_persona_command` scans the transcript for phrases like “switch to friendly voice”.
   - Successful matches activate the new persona via the PersonaManager and short-circuit the turn.

4. **State Machine Execution**
   - `COMPOSING`: Router selects the LLM (Ollama default, Gemini when tools are allowed).
   - `SPEAKING`: Kokoro TTS plays the response with persona-selected voice params.
   - `IDLE`: System resets to await the next wake word.

## Persona Manager Integration

- `Orchestrator` receives a `PersonaManager` instance and publishes persona profiles alongside every `COMPOSING`/`SPEAKING` update.
- Kokoro’s voice is set whenever a persona changes, so no per-turn overrides are needed.
- The UI card mirrors the active persona, keeping operators in sync with the voice they hear.

## Memory & Tooling

- `MemoryStore` is invoked with `guard` → `write` semantics, allowing for safe persistence of LLM-suggested memories.
- Tool calls are stubbed out by `ToolExecutorStub` today; swap in a real executor for production automations.

## Extensibility Points

- Add new system states by extending the `State` literal in `jarvis.orchestrator.events` (V2 already adds `PERSONA`).
- Override router logic if you want persona-specific provider selection (e.g., fun persona might default to a humour-tuned model).
- Persona commands can be widened—teach `PersonaManager.match_command` new aliases or intent parsing for richer control phrases.
