# Audio Pipeline

The audio path covers capture, buffering, transcription, and playback cues. Understanding it helps when tuning latency or chasing glitches.

## Capture

- **Device**: `sounddevice.InputStream` in float32 mode for stability across macOS audio drivers.
- **Frame Size**: `frame_ms` (default 30 ms) → 480 samples per frame at 16 kHz.
- **Energy VAD**: Simple mean-absolute-energy threshold (`energy_threshold=500`). Swap in a smarter detector by passing a callable to `AudioCapture`.
- **Ring Buffer**: 12 s of PCM retained for reference attachments sent to the orchestrator.

## Transcription Queue

- Frames enqueue via `TranscriptionEngine.enqueue_audio(pcm, ts, vad, force=False)`.
- Runtime stops enqueuing when `_listening_active` is false.
- Silence detection increments `_silence_duration` only when VAD says “no speech”. At 0.6 s it triggers a forced flush.

## Cheetah Specifics

- Converts PCM16 → float32 per frame before calling `pvcheetah.process`.
- Emits partial transcripts immediately; finals arrive via `flush()` when Cheetah senses an endpoint or the runtime enforces one.
- Range metadata is tracked in milliseconds relative to the current segment.

## Vosk Specifics

- Feeds raw PCM16 into `KaldiRecognizer.AcceptWaveform`.
- Non-final frames surface via `PartialResult`; finals via `Result` or `FinalResult` during forced flush.
- Sample counters estimate segment duration, keeping UI range displays consistent with Cheetah.

## Switching Logic

- Language detection (`langdetect`) analyses final transcripts.
- If Indic languages dominate, `_swap_transcriber("vosk")` is invoked inside a lock.
- When English confidence returns (>0.6), the runtime quietly reinitialises Cheetah for the next turn.

## Playback

- Kokoro returns streaming WAV bytes which are decoded via `soundfile` and played synchronously with `sounddevice.play(..., blocking=True)` to avoid overlapping speech.

Tune the inputs, thresholds, and fallback order to match your deployment environment; the pipeline was built to make those tweaks straightforward.
