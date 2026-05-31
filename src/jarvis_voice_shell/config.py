"""Configuration for JARVIS Voice Shell.

Sources (lowest to highest priority):
    1. Defaults (this module)
    2. Environment variables (JARVIS_*)
    3. CLI flags (applied by cli.py)

All paths use pathlib for Windows compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_cache_dir() -> Path:
    """Platform-appropriate cache directory for audio artifacts."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "jarvis-voice-shell" / "audio_cache"
    return Path.home() / ".cache" / "jarvis-voice-shell"


@dataclass
class Config:
    """Voice shell configuration. Immutable-ish; use replace() for overrides."""

    # -- Audio input ------------------------------------------------------------
    input_device_index: int | None = None
    """PyAudio device index. None = auto-select via AudioDeviceManager priority."""

    input_device_name: str | None = None
    """Friendly name override for display/logging."""

    sample_rate: int = 16000
    """Input sample rate (16 kHz — standard for STT engines)."""

    channels: int = 1
    """Mono input."""

    chunk_size: int = 1024
    """Frames per PyAudio buffer read."""

    # -- Push-to-talk -----------------------------------------------------------
    ptt_key: str = "`"
    """Keyboard shortcut to activate listening. Uses keyboard library syntax."""

    ptt_enabled: bool = True
    """Disable to use VAD or always-on mode."""

    vad_energy_threshold: int = 300
    """RMS threshold for always-on VAD speech detection."""

    vad_start_frames: int = 1
    """Consecutive loud frames needed to begin a speech segment."""

    vad_end_silence_ms: int = 700
    """Trailing silence needed to end a speech segment."""

    vad_pre_roll_ms: int = 300
    """Audio retained before VAD triggers, to avoid clipped starts."""

    # -- Bridge -----------------------------------------------------------------
    hermes_bridge_url: str = "http://127.0.0.1:8642/v1/chat/completions"
    """Hermes HTTP endpoint (streaming-capable OpenAI-compatible API)."""

    hermes_api_key: str = field(default_factory=lambda: os.getenv("API_SERVER_KEY", ""))
    """Bearer token for Hermes Gateway API Server; defaults to API_SERVER_KEY."""

    bridge_timeout: float = 60.0
    """HTTP timeout in seconds for bridge connections."""

    bridge_stream: bool = True
    """Request streaming response from Hermes."""

    bridge_max_tokens: int = field(default_factory=lambda: int(os.getenv("BRIDGE_MAX_TOKENS", "512")))
    """Maximum assistant tokens requested through the HTTP bridge."""

    # -- TTS --------------------------------------------------------------------
    tts_engine: str = "edge"
    """TTS backend: 'edge' (v1), 'system' (future), 'piper' (future)."""

    tts_voice: str = "en-GB-RyanNeural"
    """Edge TTS voice. en-GB-RyanNeural = British male, dry, authoritative."""

    tts_rate: str = "+0%"
    """Speech rate modifier for edge-tts (e.g. '+10%', '-5%')."""

    tts_output_device_index: int | None = None
    """PyAudio output device index. None = auto-select (headset priority)."""

    tts_playback_enabled: bool = True
    """Whether to play audio after synthesis. Set False to synthesize-only (dry run)."""

    # -- STT (stub — will be activated with `stt` extra) ------------------------
    stt_engine: str = "stub"
    """STT backend: 'stub' (always returns placeholder), 'whisper' (future)."""

    stt_model: str = "tiny"
    """Whisper model size when STT is enabled: tiny, base, small, medium, large."""

    # -- Paths ------------------------------------------------------------------
    cache_dir: Path = field(default_factory=_default_cache_dir)

    log_dir: Path | None = None
    """Directory for latency JSONL logs. Defaults to cache_dir / 'latency_logs'."""

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.log_dir is None:
            self.log_dir = self.cache_dir / "latency_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- Env loading ------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Config:
        """Load overrides from JARVIS_* environment variables."""
        kwargs = {}
        env_map = {
            "JARVIS_INPUT_DEVICE": ("input_device_index", int),
            "JARVIS_SAMPLE_RATE": ("sample_rate", int),
            "JARVIS_PTT_KEY": ("ptt_key", str),
            "JARVIS_BRIDGE_URL": ("hermes_bridge_url", str),
            "HERMES_BRIDGE_URL": ("hermes_bridge_url", str),
            "API_SERVER_KEY": ("hermes_api_key", str),
            "BRIDGE_MAX_TOKENS": ("bridge_max_tokens", int),
            "JARVIS_TTS_ENGINE": ("tts_engine", str),
            "JARVIS_TTS_VOICE": ("tts_voice", str),
            "JARVIS_TTS_RATE": ("tts_rate", str),
            "JARVIS_TTS_OUTPUT_DEVICE": ("tts_output_device_index", int),
            "JARVIS_STT_ENGINE": ("stt_engine", str),
            "JARVIS_STT_MODEL": ("stt_model", str),
            "JARVIS_LOG_DIR": ("log_dir", Path),
            "JARVIS_VAD_ENERGY_THRESHOLD": ("vad_energy_threshold", int),
            "JARVIS_VAD_START_FRAMES": ("vad_start_frames", int),
            "JARVIS_VAD_END_SILENCE_MS": ("vad_end_silence_ms", int),
            "JARVIS_VAD_PRE_ROLL_MS": ("vad_pre_roll_ms", int),
        }
        for env_var, (attr, cast) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                kwargs[attr] = cast(val)
        return cls(**kwargs)

    def replace(self, **overrides) -> Config:
        """Return a new Config with selective overrides (like dataclasses.replace)."""
        current = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        current.update(overrides)
        return Config(**current)
