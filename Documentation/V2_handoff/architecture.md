# Architecture Overview

The Creo voice agent is a pipeline of loosely coupled services coordinated by the runtime in `jarvis.main`. Each piece is independently swappable, but the defaults ship with the flow below.

## High-Level Flow

1. **AudioCapture (`jarvis.audio.capture`)**
   - Pulls 30 ms PCM16 frames at 16 kHz from CoreAudio.
   - Maintains a 12 s ring buffer for post-turn context.
   - Emits VAD-tagged frames onto the async loop.

2. **Transcription Engines (`jarvis.transcription.*`)**
   - `CheetahStream` receives PCM, performs PCM16 → float32 conversion, and streams partials/finals.
   - `VoskStream` consumes the same queue, returning interim/complete JSON results from KaldiRecognizer.
   - The runtime chooses an engine at startup and can rebuild it at runtime when language detection requests a swap.

3. **Runtime Orchestration (`jarvis.main.Runtime`)**
   - Gated audio loop enqueues frames only while `_listening_active` is true.
   - Silence timeout (0.6 s) forces a transcription flush for responsive turn taking.
   - Persona-manager hooks intercept voice/persona commands before the LLM sees them.

4. **Orchestrator (`jarvis.orchestrator.state_machine`)**
   - Serialises turn handling to protect shared memory.
   - Publishes state transitions (`LISTENING`, `COMPOSING`, `SPEAKING`, etc.) to the floating UI bridge.
   - Ensures Kokoro TTS uses persona-selected voice presets for every utterance.

5. **LLM Providers (`jarvis.llm.providers.*`)**
   - Ollama and Gemini clients now emit the shared `SYSTEM_PROMPT` so every reply stays on-brand (“Creo”, ≤ 10 words, dry tone).
   - Routing logic remains simple: default to Ollama unless the context asks for tool access requiring Gemini.

6. **UI (`jarvis.ui.static`)**
   - WebSocket bridge relays state payloads straight to the floating panel.
   - Persona+voice metadata is surfaced so operators can see which profile is live.

## Key Threads

- **Audio Loop** — Runs forever, but is entirely async and guarded, so switching transcribers or personas never blocks.
- **Transcription Loop** — Each engine owns its own async generator; replacing an engine cancels its task and spins up a fresh one.
- **Wake Word Loop** — Optional, but when active it flips `_listening_active` and publishes UI cues.

## Error Handling and Fallback

- Engine construction happens via `build_transcriber(settings, preferred)` which iterates the candidate list (`cheetah` → `vosk`).
- Failures are logged with `transcription.engine.failed` tags; the runtime immediately tries the next engine.
- During operation, exceptions inside the transcription loop trigger an automatic swap to Vosk (unless it is already active) without taking down the runtime.

With these pieces you can extend, replace, or mock any subsystem without rewriting the rest of the stack.
