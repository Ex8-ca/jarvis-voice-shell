"""JARVIS Voice Shell — realtime push-to-talk voice interface for Hermes AI Agent.

Architecture:
    Mic (PTT) → STT stub → Hermes HTTP Bridge (streaming) → Edge TTS → Headset

Key design properties:
    - Push-to-talk for deterministic activation
    - Deterministic barge-in via keyboard interrupt (outside model)
    - Streaming HTTP bridge from day one (token-level latency matters)
    - Edge TTS v1 with acknowledged network dependency
    - Bluetooth gaming headset as primary audio device
    - Local STT deferred to v2 (stub present)
    - No dashboard in v1
    - pathlib-friendly paths throughout
"""

__version__ = "0.1.0"
__all__ = [
    "AudioDeviceManager",
    "HermesBridge",
    "EchoBridge",
    "SSEParser",
    "TTSEngine",
    "LatencyTracker",
    "LatencyLogger",
    "AudioRecorder",
    "RecordingResult",
    "STTEngine",
    "TranscriptionResult",
    "Config",
    "cli_main",
]

from .audio_devices import AudioDeviceManager
from .bridge import EchoBridge, HermesBridge, SSEParser
from .config import Config
from .latency import LatencyLogger, LatencyTracker
from .recorder import AudioRecorder, RecordingResult
from .stt import STTEngine, TranscriptionResult
from .tts import TTSEngine


def cli_main():
    """Entry point for `jarvis-voice` console script."""
    from .cli import main
    main()
