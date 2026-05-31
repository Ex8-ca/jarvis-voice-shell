"""Speech-to-text helpers for JARVIS Voice Shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config


class STTError(Exception):
    """Raised when transcription fails."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    backend: str = "unknown"


class STTEngine:
    """Small STT facade.

    Backends:
    - ``stub``: deterministic placeholder, useful for shell tests.
    - ``whisper``: local OpenAI Whisper package if installed.
    """

    def __init__(self, config: Config, whisper_module: Any | None = None):
        self._config = config
        self._whisper_module = whisper_module
        self._model: Any | None = None

    def transcribe_file(self, path: Path) -> TranscriptionResult:
        path = Path(path)
        if not path.exists():
            raise STTError(f"Audio file not found: {path}")

        if self._config.stt_engine == "stub":
            return TranscriptionResult(
                text=f"[stub transcript from {path.name}]",
                backend="stub",
            )
        if self._config.stt_engine == "whisper":
            return self._transcribe_whisper(path)
        raise STTError(f"Unknown STT engine: {self._config.stt_engine}")

    def _load_whisper(self) -> Any:
        if self._whisper_module is not None:
            return self._whisper_module
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise STTError(
                "Whisper backend requires the openai-whisper package. Install it or use stt_engine='stub'."
            ) from exc
        return whisper

    def _transcribe_whisper(self, path: Path) -> TranscriptionResult:
        whisper = self._load_whisper()
        if self._model is None:
            self._model = whisper.load_model(self._config.stt_model)
        try:
            result = self._model.transcribe(str(path), fp16=False)
        except Exception as exc:  # pragma: no cover - model/runtime-specific
            raise STTError(f"Whisper transcription failed: {exc}") from exc
        return TranscriptionResult(
            text=(result.get("text") or "").strip(),
            language=result.get("language"),
            backend="whisper",
        )
