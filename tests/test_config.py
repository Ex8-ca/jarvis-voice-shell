"""Tests for config.py."""

import os
from pathlib import Path

from jarvis_voice_shell.config import Config


class TestConfigDefaults:
    """Default values should be sensible."""

    def test_default_ptt_enabled(self):
        cfg = Config()
        assert cfg.ptt_enabled is True

    def test_default_sample_rate(self):
        cfg = Config()
        assert cfg.sample_rate == 16000

    def test_default_bridge_url(self):
        cfg = Config()
        assert cfg.hermes_bridge_url == "http://127.0.0.1:8642/v1/chat/completions"

    def test_default_bridge_max_tokens_is_voice_sized(self):
        cfg = Config()
        assert cfg.bridge_max_tokens == 512

    def test_default_hermes_api_key_reads_api_server_key(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "local-secret")
        cfg = Config()
        assert cfg.hermes_api_key == "local-secret"

    def test_default_tts_voice_is_british_male(self):
        cfg = Config()
        assert "Ryan" in cfg.tts_voice
        assert "GB" in cfg.tts_voice

    def test_cache_dir_is_path(self):
        cfg = Config()
        assert isinstance(cfg.cache_dir, Path)

    def test_cache_dir_exists(self):
        cfg = Config()
        assert cfg.cache_dir.exists()

    def test_log_dir_defaults_to_cache_subdir(self):
        cfg = Config()
        assert isinstance(cfg.log_dir, Path)
        assert cfg.log_dir.parent == cfg.cache_dir
        assert cfg.log_dir.name == "latency_logs"

    def test_log_dir_exists_after_init(self):
        cfg = Config()
        assert cfg.log_dir.exists()


    def test_default_ptt_key_is_backtick_for_mouse_keybind(self):
        cfg = Config()
        assert cfg.ptt_key == "`"

    def test_default_vad_settings_support_always_on_mode(self):
        cfg = Config()
        assert cfg.vad_energy_threshold == 300
        assert cfg.vad_start_frames == 1
        assert cfg.vad_end_silence_ms == 700


class TestConfigFromEnv:
    """Environment variable loading."""

    def test_loads_input_device(self, monkeypatch):
        monkeypatch.setenv("JARVIS_INPUT_DEVICE", "3")
        cfg = Config.from_env()
        assert cfg.input_device_index == 3

    def test_loads_bridge_url(self, monkeypatch):
        monkeypatch.setenv("JARVIS_BRIDGE_URL", "http://localhost:9999/v1")
        cfg = Config.from_env()
        assert cfg.hermes_bridge_url == "http://localhost:9999/v1"

    def test_loads_api_server_key_for_hermes_bridge_auth(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "voice-key")
        cfg = Config.from_env()
        assert cfg.hermes_api_key == "voice-key"

    def test_loads_bridge_max_tokens(self, monkeypatch):
        monkeypatch.setenv("BRIDGE_MAX_TOKENS", "768")
        cfg = Config.from_env()
        assert cfg.bridge_max_tokens == 768

    def test_loads_tts_voice(self, monkeypatch):
        monkeypatch.setenv("JARVIS_TTS_VOICE", "en-US-AriaNeural")
        cfg = Config.from_env()
        assert cfg.tts_voice == "en-US-AriaNeural"

    def test_loads_tts_rate(self, monkeypatch):
        monkeypatch.setenv("JARVIS_TTS_RATE", "+10%")
        cfg = Config.from_env()
        assert cfg.tts_rate == "+10%"

    def test_loads_ptt_key(self, monkeypatch):
        monkeypatch.setenv("JARVIS_PTT_KEY", "ctrl+alt+j")
        cfg = Config.from_env()
        assert cfg.ptt_key == "ctrl+alt+j"

    def test_loads_log_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path / "custom_logs"))
        cfg = Config.from_env()
        assert cfg.log_dir == tmp_path / "custom_logs"

    def test_missing_env_vars_use_defaults(self, monkeypatch):
        # Ensure env vars are clean
        for var in list(os.environ):
            if var.startswith("JARVIS_"):
                monkeypatch.delenv(var, raising=False)
        cfg = Config.from_env()
        assert cfg.sample_rate == 16000  # default


class TestConfigReplace:
    """Immutable-style replace()."""

    def test_replace_returns_new_instance(self):
        cfg = Config()
        new = cfg.replace(sample_rate=48000)
        assert new is not cfg
        assert new.sample_rate == 48000
        assert cfg.sample_rate == 16000  # original unchanged

    def test_replace_multiple_fields(self):
        cfg = Config()
        new = cfg.replace(sample_rate=44100, tts_voice="en-US-JennyNeural")
        assert new.sample_rate == 44100
        assert new.tts_voice == "en-US-JennyNeural"

    def test_replace_preserves_other_fields(self):
        cfg = Config(ptt_key="ctrl+j")
        new = cfg.replace(tts_voice="test")
        assert new.ptt_key == "ctrl+j"

    def test_replace_log_dir(self):
        cfg = Config()
        new = cfg.replace(log_dir=Path("/tmp/test_logs"))
        assert new.log_dir == Path("/tmp/test_logs")
        assert cfg.log_dir != Path("/tmp/test_logs")
