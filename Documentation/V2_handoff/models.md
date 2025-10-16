# Models & Engines

This release leans on a small set of focused models. The table below summarises each component, its role, where it lives, and how to tune it.

| Component | Purpose | Location | Notes |
| --- | --- | --- | --- |
| **Picovoice Cheetah** | Primary low-latency ASR for English | `jarvis/transcription/cheetah.py` | Requires `CHEETAH_ACCESS_KEY`. Optional `CHEETAH_MODEL_PATH` can point to a custom `.pv` model if you need domain language. |
| **Vosk (vosk-model-small-en-in)** | Hinglish-friendly ASR fallback | `jarvis/transcription/vosk.py` | Drop the model under `Voice system/models/`. Swap `VOSK_MODEL_PATH` in `.env` for larger checkpoints. |
| **langdetect** | Lightweight language classifier | Used inside `Runtime._detect_languages` | Zero config; if you see noisy results, pre-clean transcripts before detection. |
| **Kokoro TTS** | Neural TTS pipeline | `jarvis/tts/kokoro.py` | Persona manager chooses presets from `VOICE_PRESETS`; align voice weights or add new entries there. |
| **Ollama (default: llama3.2)** | Local LLM inference | External service (`settings.llm.ollama_host`) | System prompt now hardcodes the Creo persona. Swap models via the Ollama server config. |
| **Gemini (gemini-1.5-pro)** | Cloud LLM with tool support | `settings.llm.gemini_api_key` | Optional; router only calls it if tools are requested in context. |

## Persona Voice Presets

`jarvis/tts/voices.py` defines reusable voice payloads for Kokoro. Personas reference these keys:

- `english_male` (default)
- `english_female`
- `fun_male`
- `fun_female`
- add new presets there before wiring them into `PERSONAS`.

## Adding a New ASR Model

1. Create a new engine under `jarvis/transcription/` implementing `TranscriptionEngine`.
2. Update `build_transcriber` in `jarvis.main` to include the new engine in the fallback list.
3. Extend `.env` + `AppSettings` with whatever knobs the engine requires.

Because the runtime already handles queue management and task swaps, most new engines only need conversion logic plus partial/final chunk emission.
