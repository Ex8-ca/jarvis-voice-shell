"""Local voice mode: VAD → Whisper → Hermes → Edge TTS (no OpenAI API).

Pipeline:
    1. Energy VAD detects speech end
    2. Recorder captures the segment
    3. Whisper transcribes locally
    4. Hermes bridge (hermes-cli) reasons
    5. Edge TTS synthesises and plays

Requires: whisper, edge-tts, sounddevice, numpy
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from .config import Config
from .stt import STTEngine
from .tts import TTSEngine
from .vad import EnergyVAD
from .bridge import HermesCliBridge


class LocalVoiceError(Exception):
    """Raised when local voice pipeline fails."""


class LocalVoiceClient:
    """Always-on local voice client.

    Uses energy VAD for turn detection, whisper for STT,
    Hermes bridge for reasoning, edge-tts for TTS.
    """

    def __init__(
        self,
        config: Config,
        on_status=None,
        bridge=None,
    ):
        self._config = config
        self._on_status = on_status or (lambda msg: None)
        self._bridge = bridge
        self._stt_engine = STTEngine(config)
        self._tts_engine = TTSEngine(config)
        self._vad = EnergyVAD(
            energy_threshold=config.vad_energy_threshold,
            start_frames=config.vad_start_frames,
            end_silence_frames=max(1, int((config.vad_end_silence_ms / 1000) * config.sample_rate / config.chunk_size)),
            pre_roll_frames=max(0, int((config.vad_pre_roll_ms / 1000) * config.sample_rate / config.chunk_size)),
        )
        self._running = False
        self._stop_event = asyncio.Event()

    def _load_whisper(self):
        """Load whisper module (lazy, on first use)."""
        try:
            import whisper
        except ImportError:
            raise LocalVoiceError(
                "whisper is not installed. Install with: pip install openai-whisper"
            )
        return whisper

    async def run(self):
        """Main loop: detect speech, transcribe, reason, speak."""
        import sounddevice as sd
        import numpy as np

        self._on_status("Local voice mode ready.")
        self._on_status(f"STT: whisper (local)")
        self._on_status(f"TTS: edge-tts ({self._config.tts_voice})")
        self._on_status(f"Brain: {self._config.hermes_bridge_url}")
        self._on_status("Listening... (energy VAD)")
        self._on_status("Ctrl+C exits.")
        self._running = True

        sd_np = self._load_sounddevice(sd)
        if sd_np is None:
            raise LocalVoiceError("Always-on VAD requires sounddevice+numpy")
        sd, np = sd_np

        device_index = (
            self._config.input_device_index
            if self._config.input_device_index is not None
            else self._find_input_device(sd)
        )
        sample_rate = int(self._config.sample_rate)
        chunk_size = int(self._config.chunk_size)
        max_chunks = max(1, int(30.0 / chunk_size * sample_rate))

        try:
            while self._running:
                segment = await self._record_vad_segment(
                    sd, np, device_index, sample_rate, chunk_size, max_chunks
                )
                if segment is None:
                    continue

                self._on_status("Transcribing...")
                text = await self._transcribe(segment)
                if not text.strip():
                    continue

                self._on_status(f"Hermes heard: {text}")
                response = await self._ask_brain(text)
                self._on_status(f"Hermes said: {response}")
                await self._speak(response)

        except Exception as exc:
            self._on_status(f"Voice error: {exc}")
            raise

    def _load_sounddevice(self, sd):
        """Load sounddevice and numpy."""
        try:
            import numpy as np
        except ImportError:
            return None
        return sd, np

    def _find_input_device(self, sd):
        """Find the best input device by name."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = dev["name"].lower()
                if any(kw in name for kw in ["headset", "xbox", "mic", "microphone", "input"]):
                    return i
        return 0

    async def _record_vad_segment(self, sd, np, device_index, sample_rate, chunk_size, max_chunks):
        """Record one speech segment using VAD."""
        q = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                return
            q.put_nowait(bytes(indata))

        segment = None
        chunks_read = 0
        peak_rms = 0

        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device_index,
            blocksize=chunk_size,
            callback=callback,
        ):
            while chunks_read < max_chunks and not self._stop_event.is_set():
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                chunks_read += 1
                peak_rms = max(peak_rms, self._rms_int16(frame))
                prior_state = self._vad.state
                segment = self._vad.process(frame)
                if prior_state != "speaking" and self._vad.state == "speaking":
                    self._on_status("Speech detected.")
                if segment is not None:
                    break

        if segment is None:
            return None
        return segment

    @staticmethod
    def _rms_int16(frame):
        """RMS amplitude for int16 mono PCM."""
        if not frame:
            return 0
        samples = len(frame) // 2
        if samples <= 0:
            return 0
        total = 0
        for i in range(0, samples * 2, 2):
            sample = int.from_bytes(frame[i : i + 2], "little", signed=True)
            total += sample * sample
        return int((total / samples) ** 0.5)

    async def _transcribe(self, audio_segment):
        """Transcribe audio segment with whisper."""
        import tempfile
        import wave

        # Write audio to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
            with wave.open(f, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self._config.sample_rate)
                wav.writeframes(audio_segment)

        try:
            whisper = self._load_whisper()
            if not hasattr(self, "_whisper_model") or self._whisper_model is None:
                self._whisper_model = whisper.load_model(self._config.stt_model)

            result = self._whisper_model.transcribe(
                str(temp_path), fp16=False, language="en",
            )
            return result.get("text", "")
        finally:
            temp_path.unlink(missing_ok=True)

    async def _ask_brain(self, transcript):
        """Send transcript to Hermes bridge."""
        if self._bridge is None:
            bridge = HermesCliBridge()
        else:
            bridge = self._bridge

        response = await bridge.send(transcript)
        return response.strip()

    async def _speak(self, text):
        """Synthesise and play with edge-tts."""
        await self._tts_engine.speak(text)

    def stop(self):
        """Stop the voice client."""
        self._running = False
        self._stop_event.set()


def _default_voice_for_jarvis():
    """Return the best edge-tts voice for a Jarvis-like tone."""
    # en-GB-RyanNeural = British male, dry, authoritative
    # en-GB-SebastianNeural = deeper British male
    # en-US-GuyNeural = American male, clear
    return "en-GB-RyanNeural"
