from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    dsn: str
    max_pool_size: int = 10


class RealTimeSTTSettings(BaseModel):
    base_url: str
    auth_token: str | None = None
    sample_rate: int = 16_000
    frame_ms: int = 30
    energy_threshold: float = 500.0
    input_device: str | int | None = None


class TranscriptionSettings(BaseModel):
    engine: Literal["cheetah", "vosk"] = "cheetah"
    cheetah_access_key: str | None = None
    cheetah_model_path: str | None = None
    cheetah_endpoint_sec: float = 0.4
    vosk_model_path: str | None = None
    vosk_model_path_hi: str | None = None
    vosk_model_path_en_in: str | None = None


class KokoroSettings(BaseModel):
    base_url: str
    api_key: str | None = None
    default_voice: str = "english_male"


class WakeWordSettings(BaseModel):
    engine: Literal["porcupine", "openwakeword"] = "porcupine"
    keyword: str = "jarvis"  # TODO: flip back to 'creo' when upgraded keyword model is ready
    porcupine_keyword_path: str | None = None
    porcupine_model_path: str | None = None
    porcupine_access_key: str | None = None


class LLMSettings(BaseModel):
    ollama_host: str = "http://localhost:11434"
    gemini_api_key: str | None = None


class TelemetrySettings(BaseModel):
    log_level: str = "INFO"
    otlp_endpoint: str | None = None
    resource_sample_seconds: int = 30


class UISettings(BaseModel):
    floating_ui_origin: str = "http://localhost:8010"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore")

    POSTGRES_DSN: str
    POSTGRES_POOL_SIZE: int = 10
    REALTIME_STT_URL: str
    REALTIME_STT_AUTH_TOKEN: str | None = None
    REALTIME_STT_SAMPLE_RATE: int = 16_000
    REALTIME_STT_FRAME_MS: int = 30
    AUDIO_ENERGY_THRESHOLD: float = 500.0
    AUDIO_INPUT_DEVICE: str | int | None = None
    TRANSCRIPTION_ENGINE: Literal["cheetah", "vosk"] = "vosk"
    CHEETAH_ACCESS_KEY: str | None = None
    CHEETAH_MODEL_PATH: str | None = None
    CHEETAH_ENDPOINT_SEC: float = 0.4
    VOSK_MODEL_PATH: str | None = None
    VOSK_MODEL_PATH_HI: str | None = None
    VOSK_MODEL_PATH_EN_IN: str | None = None
    KOKORO_API_URL: str
    KOKORO_API_KEY: str | None = None
    KOKORO_DEFAULT_VOICE: str = "english_male"
    WAKEWORD_ENGINE: Literal["porcupine", "openwakeword"] = "porcupine"
    WAKEWORD_KEYWORD: str = "jarvis"
    PORCUPINE_KEYWORD_PATH: str | None = None
    PORCUPINE_MODEL_PATH: str | None = None
    PORCUPINE_ACCESS_KEY: str | None = None
    OLLAMA_HOST: str = "http://localhost:11434"
    GEMINI_API_KEY: str | None = None
    LOG_LEVEL: str = "INFO"
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    RESOURCE_SAMPLE_SECONDS: int = 30
    FLOATING_UI_ORIGIN: str = "http://localhost:8010"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(dsn=self.POSTGRES_DSN, max_pool_size=self.POSTGRES_POOL_SIZE)

    @staticmethod
    def _coerce_device(device: str | int | None) -> str | int | None:
        if isinstance(device, str):
            trimmed = device.strip()
            if not trimmed:
                return None
            if trimmed.isdigit():
                return int(trimmed)
            return trimmed
        return device

    @property
    def realtime_stt(self) -> RealTimeSTTSettings:
        return RealTimeSTTSettings(
            base_url=self.REALTIME_STT_URL,
            auth_token=self.REALTIME_STT_AUTH_TOKEN,
            sample_rate=self.REALTIME_STT_SAMPLE_RATE,
            frame_ms=self.REALTIME_STT_FRAME_MS,
            energy_threshold=self.AUDIO_ENERGY_THRESHOLD,
            input_device=self._coerce_device(self.AUDIO_INPUT_DEVICE),
        )

    @property
    def kokoro(self) -> KokoroSettings:
        return KokoroSettings(
            base_url=self.KOKORO_API_URL,
            api_key=self.KOKORO_API_KEY,
            default_voice=self.KOKORO_DEFAULT_VOICE,
        )

    @property
    def wakeword(self) -> WakeWordSettings:
        return WakeWordSettings(
            engine=self.WAKEWORD_ENGINE,
            keyword=self.WAKEWORD_KEYWORD,
            porcupine_keyword_path=self.PORCUPINE_KEYWORD_PATH,
            porcupine_model_path=self.PORCUPINE_MODEL_PATH,
            porcupine_access_key=self.PORCUPINE_ACCESS_KEY,
        )

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(ollama_host=self.OLLAMA_HOST, gemini_api_key=self.GEMINI_API_KEY)

    @property
    def telemetry(self) -> TelemetrySettings:
        return TelemetrySettings(
            log_level=self.LOG_LEVEL,
            otlp_endpoint=self.OTEL_EXPORTER_OTLP_ENDPOINT,
            resource_sample_seconds=self.RESOURCE_SAMPLE_SECONDS,
        )

    @property
    def ui(self) -> UISettings:
        return UISettings(floating_ui_origin=self.FLOATING_UI_ORIGIN)

    @property
    def transcription(self) -> TranscriptionSettings:
        return TranscriptionSettings(
            engine=self.TRANSCRIPTION_ENGINE,
            cheetah_access_key=self.CHEETAH_ACCESS_KEY,
            cheetah_model_path=self.CHEETAH_MODEL_PATH,
            cheetah_endpoint_sec=self.CHEETAH_ENDPOINT_SEC,
            vosk_model_path=self.VOSK_MODEL_PATH,
            vosk_model_path_hi=self.VOSK_MODEL_PATH_HI,
            vosk_model_path_en_in=self.VOSK_MODEL_PATH_EN_IN,
        )


@functools.lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    return AppSettings()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


__all__ = ["AppSettings", "load_settings", "project_root"]
