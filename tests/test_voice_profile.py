"""Tests for voice-only prompt wrapping and TTS sanitizing."""

from jarvis_voice_shell.voice_profile import (
    is_status_query,
    sanitize_for_speech,
    spoken_failure_message,
    wrap_for_voice,
)


def test_wrap_for_voice_adds_spoken_style_without_losing_transcript():
    wrapped = wrap_for_voice("check the server")

    assert "spoken JARVIS voice session" in wrapped
    assert "Be brief" in wrapped
    assert "check the server" in wrapped


def test_sanitize_for_speech_removes_markdown_paths_and_symbols():
    raw = "**Done** — see `C:/Users/Example/test-file.py` and https://example.com/a-b. \"OK\" / fine"

    spoken = sanitize_for_speech(raw)

    assert "**" not in spoken
    assert "`" not in spoken
    assert "C:/" not in spoken
    assert "https://" not in spoken
    assert '"' not in spoken
    assert "/" not in spoken
    assert "Done" in spoken
    assert "file path" in spoken


def test_sanitize_for_speech_clips_long_output():
    spoken = sanitize_for_speech("word " * 300, max_chars=80)

    assert len(spoken) < 140
    assert "more detail on screen" in spoken


def test_status_query_detection():
    assert is_status_query("Jarvis, are you still there?") is True
    assert is_status_query("system status") is True
    assert is_status_query("turn on the lights") is False


def test_spoken_failure_messages_are_short_and_natural():
    assert spoken_failure_message("Hermes CLI failed") == "Hermes bridge failure, sir. I am still listening."
    assert spoken_failure_message("request timeout") == "Model timeout, sir. I am still listening."
