# Resource Management

Keeping Creo lightweight on an M3 Pro was a design goal for V2. The notes below explain how the runtime manages memory/CPU and what you can tune if things drift.

## Baseline Footprint

- **AudioCapture** uses NumPy buffers sized to the ring window (≈ 3 MB at 16 kHz/16-bit mono for 12 s).
- **Cheetah** runs fully on CPU; the Picovoice engine allocates a small resident model (≤ 30 MB) and processes 30 ms hops.
- **Vosk** consumes more RAM (~50 MB for `vosk-model-small-en-in`) but loads lazily and is only active when selected.
- **LLMs** live outside this repo (Ollama/Gemini); only HTTP clients are retained locally.
- **Kokoro TTS** streams audio chunks and never stores full outputs in memory; playback uses `sounddevice` in blocking mode to prevent accumulation.

## Tunables

| Setting | Default | Impact |
| --- | --- | --- |
| `REALTIME_STT_SAMPLE_RATE` | 16000 | Lowering reduces bandwidth and CPU, but both Cheetah and Vosk expect 16 kHz. |
| `REALTIME_STT_FRAME_MS` | 30 | Controls AudioCapture chunk size; larger frames increase latency, smaller frames increase scheduling overhead. |
| `CHEETAH_ENDPOINT_SEC` | 0.4 | Controls how quickly Cheetah flushes final segments. Larger values reduce CPU at the cost of slower turn boundaries. |
| `VOSK_MODEL_PATH` | small model | Swap to `vosk-model-en-us-0.22` for better accuracy at the expense of 1 GB+ disk/RAM. |
| `RESOURCE_SAMPLE_SECONDS` | 30 | Governs how often system metrics are published to the UI. Raising this to 60+ reduces logging chatter. |

## Runtime Safeguards

- **Switch Lock**: `_switch_lock` in `Runtime` serialises transcriber swaps so you never instantiate two heavy engines simultaneously.
- **Ensure Hook**: `_ensure_transcriber_ready` rebuilds an engine if it goes missing, but only while the lock is held, preventing runaway rebuild loops.
- **Silence Timeout**: 0.6 s cut-off keeps the transcription queue from growing indefinitely when a microphone is left open.
- **Langdetect Backoff**: Language detection only runs on final transcripts longer than 4 characters to avoid spinning CPU on noise.

## Monitoring Tips

- The floating UI’s “Diagnostics” state displays CPU %, RAM %, and RSS MB. Use it periodically, especially after installing new models.
- For deeper insight, enable OTLP to push telemetry into Grafana/Tempo; the hooks remain from V1 (see `LOG_LEVEL`/`OTEL_EXPORTER_OTLP_ENDPOINT`).

## Memory Hygiene

- Make sure to shut down gracefully (`/manual/wake/stop` or the shutdown command). The runtime awaits all async tasks and closes engines so native buffers are released.
- If you hot swap models often, clean the `__pycache__` directories to avoid stale artifacts stacking up.

With these practices, the agent idles below 150 MB RSS and peaks under 400 MB during active turns on an M3 Pro.
