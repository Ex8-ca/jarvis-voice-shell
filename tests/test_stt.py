"""Tests for stt.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_voice_shell.config import Config
from jarvis_voice_shell.stt import STTEngine, STTError


def test_stub_transcribes_existing_file(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")
    result = STTEngine(Config(stt_engine="stub")).transcribe_file(audio)
    assert result.backend == "stub"
    assert "clip.wav" in result.text


def test_transcribe_missing_file_raises(tmp_path):
    with pytest.raises(STTError, match="Audio file not found"):
        STTEngine(Config()).transcribe_file(tmp_path / "missing.wav")


def test_unknown_backend_raises(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")
    with pytest.raises(STTError, match="Unknown STT engine"):
        STTEngine(Config(stt_engine="bogus")).transcribe_file(audio)


class FakeWhisperModel:
    def __init__(self):
        self.path = None

    def transcribe(self, path, fp16=False):
        self.path = path
        return {"text": " hello sir ", "language": "en"}


class FakeWhisperModule:
    def __init__(self):
        self.loaded = None
        self.model = FakeWhisperModel()

    def load_model(self, name):
        self.loaded = name
        return self.model


def test_whisper_backend_uses_configured_model(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")
    fake = FakeWhisperModule()

    result = STTEngine(
        Config(stt_engine="whisper", stt_model="tiny"),
        whisper_module=fake,
    ).transcribe_file(audio)

    assert fake.loaded == "tiny"
    assert fake.model.path == str(audio)
    assert result.text == "hello sir"
    assert result.language == "en"
    assert result.backend == "whisper"
