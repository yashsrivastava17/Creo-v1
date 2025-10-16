ASR Tester (Standalone)

Lightweight CLI + Web UI to exercise local transcription without running the full service.

Features
- Switch between Cheetah and Vosk (hi / en-IN) at runtime
- Start/stop capture
- Live audio level meter + partial/final transcripts
- CPU usage sampling

Env it uses (falls back to sensible defaults)
- AUDIO_INPUT_DEVICE (e.g., 0)
- REALTIME_STT_SAMPLE_RATE (default 16000)
- CHEETAH_ACCESS_KEY, CHEETAH_MODEL_PATH, CHEETAH_ENDPOINT_SEC, CHEETAH_AUTO_PUNCT
- VOSK_MODEL_PATH_EN_IN, VOSK_MODEL_PATH_HI

Run
  uvicorn tools.asr_tester.main:app --reload --port 8099

Then open http://localhost:8099

CLI (optional)
  python -m tools.asr_tester.main --help

