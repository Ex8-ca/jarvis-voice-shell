"""OpenAI realtime voice mode tests."""

from __future__ import annotations

import math

import pytest
from click.testing import CliRunner

from jarvis_voice_shell.cli import main
from jarvis_voice_shell.openai_voice import (
    MissingOpenAIAPIKey,
    OpenAIRealtimeVoiceClient,
    OpenAIRealtimeVoiceConfig,
    _mono_to_output_channels,
    _pull_ordered_audio,
    _realtime_headers,
    _should_cancel_response,
    _session_update_payload,
    _speak_text_payload,
    _websocket_connect_kwargs,
    _websocket_headers_kw,
)


def test_openai_voice_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingOpenAIAPIKey, match="OPENAI_API_KEY"):
        OpenAIRealtimeVoiceConfig.from_env(env_files=[])


def test_openai_voice_config_loads_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test-from-file\n", encoding="utf-8")

    cfg = OpenAIRealtimeVoiceConfig.from_env(env_files=[env_file])

    assert cfg.api_key == "sk-test-from-file"


def test_cli_exposes_openai_voice_command():
    result = CliRunner().invoke(main, ["openai-voice", "--help"])

    assert result.exit_code == 0
    assert "OpenAI Realtime" in result.output
    assert "--model" in result.output
    assert "--input-device" in result.output


def test_realtime_headers_use_ga_shape_without_beta_header():
    headers = _realtime_headers("sk-test")

    assert headers == {"Authorization": "Bearer sk-test"}
    assert "OpenAI-Beta" not in headers


def test_default_realtime_model_is_small_voice_layer():
    cfg = OpenAIRealtimeVoiceConfig(api_key="sk-test")

    assert cfg.model == "gpt-realtime-mini"


def test_session_update_payload_uses_ga_realtime_shape():
    cfg = OpenAIRealtimeVoiceConfig(api_key="sk-test", model="gpt-realtime-2", voice="ash")

    payload = _session_update_payload(cfg)

    assert payload["type"] == "session.update"
    assert payload["session"]["type"] == "realtime"
    assert payload["session"]["model"] == "gpt-realtime-2"
    assert payload["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert payload["session"]["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert payload["session"]["audio"]["output"]["voice"] == "ash"
    assert payload["session"]["audio"]["input"]["turn_detection"]["create_response"] is False
    assert payload["session"]["audio"]["input"]["transcription"]["model"] == "gpt-4o-mini-transcribe"


def test_speak_text_payload_keeps_openai_as_mouth_not_brain():
    payload = _speak_text_payload("Hermes response, sir.")

    assert payload["type"] == "response.create"
    assert payload["response"]["instructions"].endswith("Hermes response, sir.")
    assert "Do not answer independently" in payload["response"]["instructions"]


def test_speech_started_only_cancels_when_response_is_active():
    assert not _should_cancel_response(response_active=False)
    assert _should_cancel_response(response_active=True)


def test_rate_convert_does_not_require_removed_audioop_module():
    cfg = OpenAIRealtimeVoiceConfig(api_key="sk-test", device_sample_rate=44100, api_sample_rate=24000)
    client = OpenAIRealtimeVoiceClient(cfg)
    one_tenth_second = (b"\x00\x00" * 4410)

    converted = client._rate_convert(one_tenth_second, 44100, 24000)

    assert len(converted) == 2400 * 2


def test_output_buffer_preserves_order_when_chunks_span_callbacks():
    import queue

    q: queue.Queue[bytes | None] = queue.Queue()
    q.put(b"abcdef")
    q.put(b"ghij")
    pending = bytearray()

    first, pending = _pull_ordered_audio(q, pending, 4)
    second, pending = _pull_ordered_audio(q, pending, 4)
    third, pending = _pull_ordered_audio(q, pending, 4)

    assert first == b"abcd"
    assert second == b"efgh"
    assert third == b"ij\x00\x00"


def test_mono_output_is_duplicated_for_stereo_device():
    mono = (b"\x01\x00" + b"\x02\x00")

    stereo = _mono_to_output_channels(mono, 2)

    assert stereo == b"\x01\x00\x01\x00\x02\x00\x02\x00"


def test_rate_convert_preserves_tone_frequency_when_upsampling_output():
    import numpy as np

    cfg = OpenAIRealtimeVoiceConfig(api_key="sk-test", device_sample_rate=48000, api_sample_rate=24000)
    client = OpenAIRealtimeVoiceClient(cfg)
    tone_hz = 1000
    duration = 0.2
    src_rate = 24000
    dst_rate = 48000
    t = np.arange(int(src_rate * duration)) / src_rate
    samples = (0.5 * 32767 * np.sin(2 * math.pi * tone_hz * t)).astype("<i2")

    converted = client._rate_convert(samples.tobytes(), src_rate, dst_rate, output=True)
    out = np.frombuffer(converted, dtype="<i2").astype(np.float32)
    freqs = np.fft.rfftfreq(len(out), 1 / dst_rate)
    peak_hz = freqs[int(np.argmax(np.abs(np.fft.rfft(out))))]

    assert len(converted) == int(dst_rate * duration) * 2
    assert peak_hz == pytest.approx(tone_hz, abs=20)
    assert out.max() > 10000


def test_websocket_connect_kwargs_include_header_and_timeout():
    def connect(uri, *, additional_headers=None, open_timeout=None, max_size=None):
        return None

    kwargs = _websocket_connect_kwargs(connect, {"Authorization": "Bearer test"})

    assert kwargs["additional_headers"] == {"Authorization": "Bearer test"}
    assert kwargs["open_timeout"] == 10
    assert kwargs["max_size"] is None


def test_websocket_headers_kw_supports_websockets_16_signature():
    def connect(uri, *, additional_headers=None, max_size=None):
        return None

    assert _websocket_headers_kw(connect, {"Authorization": "Bearer test"}) == {
        "additional_headers": {"Authorization": "Bearer test"}
    }


def test_cli_openai_voice_reports_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "userprofile"))

    result = CliRunner().invoke(main, ["openai-voice", "--dry-run"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output
