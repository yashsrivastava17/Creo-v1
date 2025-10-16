# Environment & `.env` Reference

Everything configurable lives in `.env`. This quick reference highlights the new transcription knobs and reminds you about persona/LLM keys.

## Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

- Wheels for `pvcheetah` and `vosk` are platform-specific; installing inside the venv keeps them isolated.
- On Apple Silicon, ensure you run an `arm64` Python (Homebrew works well).

## Transcription Settings

| Variable | Default | Description |
| --- | --- | --- |
| `TRANSCRIPTION_ENGINE` | `cheetah` | Preferred primary engine. The runtime still tries both and falls back as needed. |
| `CHEETAH_ACCESS_KEY` | _(none)_ | Required. Grab it from your Picovoice console. |
| `CHEETAH_MODEL_PATH` | _(none)_ | Optional path to a `.pv` custom language model. Leave blank for default English. |
| `CHEETAH_ENDPOINT_SEC` | `0.4` | Soft endpoint length in seconds. |
| `VOSK_MODEL_PATH` | `Voice system/models/vosk-model-small-en-in` | Download the model and keep this path up to date. |

## Audio + Wake Word

| Variable | Default | Description |
| --- | --- | --- |
| `REALTIME_STT_SAMPLE_RATE` | 16000 | Keep at 16 kHz for best ASR accuracy. |
| `REALTIME_STT_FRAME_MS` | 30 | Frame size handed to AudioCapture. |
| `WAKEWORD_ENGINE` | `porcupine` | Wake engine selector. |
| `PORCUPINE_*` | _(paths/keys)_ | Provide the correct `.ppn`, `.pv`, and access key for Porcupine if used. |

## Personas & TTS

| Variable | Default | Description |
| --- | --- | --- |
| `KOKORO_API_URL` | http endpoint | Where Kokoro or your TTS proxy is hosted. |
| `KOKORO_API_KEY` | _(optional)_ | Add if your TTS server enforces auth. |
| `KOKORO_DEFAULT_VOICE` | `english_male` | Seed persona voice. Persists only during runtime; voice swaps are in-memory. |

## LLM Providers

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Local LLM server. |
| `GEMINI_API_KEY` | _(optional)_ | Required if you want Gemini fallbacks/tooling. |

## Persona Awareness

- Persona state is not persisted. If you want a different default persona, edit `DEFAULT_PERSONA` in `jarvis/persona.py`.
- The floating UI always reflects the active persona, so no additional .env wiring is needed.

Keep `.env.local` for machine-specific overrides to avoid accidentally committing secrets.
