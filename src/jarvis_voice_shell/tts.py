"""Edge TTS wrapper for JARVIS Voice Shell.

Uses edge-tts (free Microsoft Edge TTS) to synthesize speech.
Saves to cache_dir as MP3 and plays via PyAudio.

Key design properties:
    - Edge TTS via subprocess CLI (deterministic, no network if cached)
    - Optional edge_tts module (Communicate API) as preferred path
    - MP3 cached in cache_dir with content-hash naming
    - Playback is optional (tts_playback_enabled=False for dry runs)
    - Interruptible/cancellable playback via cancel_playback()
    - No secrets hardcoded — voice/rate come from Config

Known limitations:
    - Requires network for first synthesis (unless cached)
    - Not streaming at the audio level (full sentence before playback)
    - No offline fallback in v1
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

from .config import Config
from .latency import TurnLatency

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Raised when TTS synthesis or playback fails."""


class TTSEngine:
    """Synthesize speech and play it through the selected output device."""

    def __init__(self, config: Config):
        self._config = config
        self._pa: object = None  # PyAudio instance (lazy)
        self._output_stream: object = None
        self._playback_task: Optional[asyncio.Task] = None
        self._cancel_requested = False
        """Currently-running playback task, for cancellation."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def speak(
        self,
        text: str,
        latency: Optional[TurnLatency] = None,
    ) -> None:
        """Synthesize text to speech and optionally play it.

        Args:
            text: The text to speak.
            latency: If provided, tts_start / tts_first_audio / tts_end are stamped.

        Raises:
            TTSError: On synthesis or playback failure.
        """
        if not text.strip():
            return

        import time

        if latency is not None:
            latency.tts_start = time.perf_counter()

        if self._config.tts_engine == "edge":
            audio_data = await self._synthesize_edge(text)
        elif self._config.tts_engine == "system":
            audio_data = await self._synthesize_system(text)
        else:
            raise TTSError(f"Unknown TTS engine: {self._config.tts_engine}")

        if latency is not None:
            latency.tts_first_audio = time.perf_counter()

        if self._config.tts_playback_enabled:
            await self._play_wav(audio_data)
        else:
            logger.debug("Playback disabled — skipping audio output.")

        if latency is not None:
            latency.tts_end = time.perf_counter()

    # ------------------------------------------------------------------
    # Edge TTS command construction
    # ------------------------------------------------------------------

    def _construct_edge_tts_command(
        self, text: str, output_path: Path,
    ) -> list[str]:
        """Build the edge-tts CLI command as a list of arguments.

        Returns a list suitable for ``asyncio.create_subprocess_exec``::

            ["edge-tts", "--voice", "...", "--rate", "...",
             "--text", "...", "--write-media", "/path/to/output.mp3"]
        """
        return [
            "edge-tts",
            "--voice", self._config.tts_voice,
            "--rate", self._config.tts_rate,
            "--text", text,
            "--write-media", str(output_path),
        ]

    # ------------------------------------------------------------------
    # Output file path (content-hash based, deterministic)
    # ------------------------------------------------------------------

    def _tts_output_path(self, text: str) -> Path:
        """Return a deterministic output path for the given text + config.

        Uses SHA-256 of (voice, rate, text) so identical inputs produce
        identical filenames — enabling cache reuse.
        """
        key = f"{self._config.tts_voice}|{self._config.tts_rate}|{text}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self._config.cache_dir / f"tts_{digest}.mp3"

    # ------------------------------------------------------------------
    # File-based synthesis (CLI subprocess)
    # ------------------------------------------------------------------

    async def synthesize_edge_to_file(self, text: str) -> Path:
        """Synthesize text to an MP3 file in cache_dir via edge-tts CLI.

        Args:
            text: The text to synthesize.

        Returns:
            Path to the synthesized MP3 file (cached).

        Raises:
            TTSError: If edge-tts CLI fails or returns non-zero.
        """
        output_path = self._tts_output_path(text)

        # Cache hit: if file already exists, skip synthesis
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.debug("TTS cache hit: %s", output_path)
            return output_path

        cmd = self._construct_edge_tts_command(text, output_path)

        logger.debug("Running edge-tts: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace")[:500]
            # Clean up partial output
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise TTSError(f"edge-tts subprocess failed: {err_text}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TTSError("edge-tts produced no output file")

        return output_path

    # ------------------------------------------------------------------
    # Synthesis: edge-tts module (Communicate API, preferred)
    # ------------------------------------------------------------------

    async def _synthesize_edge(self, text: str) -> bytes:
        """Synthesize via edge-tts module or subprocess. Returns WAV bytes."""
        try:
            import edge_tts
        except ImportError:
            return await self._synthesize_edge_subprocess(text)

        # Try in-memory Communicate first (avoids temp file I/O)
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self._config.tts_voice,
                rate=self._config.tts_rate,
            )
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            if not chunks:
                raise TTSError("Edge TTS produced no audio data")

            mp3_data = b"".join(chunks)
            # Also save to cache for future reuse
            output_path = self._tts_output_path(text)
            try:
                output_path.write_bytes(mp3_data)
            except Exception:
                logger.debug("Failed to cache TTS output", exc_info=True)

            return await self._mp3_to_wav(mp3_data)

        except AttributeError:
            logger.debug(
                "edge-tts Communicate not available, falling back to subprocess",
            )
            return await self._synthesize_edge_subprocess(text)

    async def _synthesize_edge_subprocess(self, text: str) -> bytes:
        """Synthesize via edge-tts CLI and return WAV bytes."""
        # Use file-based synthesis for caching
        mp3_path = await self.synthesize_edge_to_file(text)
        mp3_data = mp3_path.read_bytes()
        return await self._mp3_to_wav(mp3_data)

    async def _mp3_to_wav(self, mp3_data: bytes) -> bytes:
        """Convert MP3 bytes to WAV via ffmpeg subprocess."""
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", "pipe:0",
            "-f", "wav",
            "-acodec", "pcm_s16le",
            "-ar", "24000",
            "-ac", "1",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=mp3_data)
        if proc.returncode != 0:
            logger.warning(
                "ffmpeg not available (returncode=%d), returning raw MP3. "
                "Install ffmpeg for WAV conversion.",
                proc.returncode,
            )
            return mp3_data
        return stdout

    async def _synthesize_system(self, text: str) -> bytes:
        """Stub for system TTS (e.g., Windows SAPI). Not implemented in v1."""
        raise TTSError("System TTS not implemented in v1. Use edge-tts.")

    # ------------------------------------------------------------------
    # Playback (PyAudio)
    # ------------------------------------------------------------------

    async def _play_wav(self, wav_data: bytes) -> None:
        """Play raw WAV bytes through PyAudio on the selected output device.

        Playback runs as an async task stored in self._playback_task so it
        can be cancelled externally via cancel_playback().
        """
        if not self._config.tts_playback_enabled:
            return

        self._cancel_requested = False
        self._playback_task = asyncio.ensure_future(
            self._play_wav_async(wav_data),
        )
        try:
            await self._playback_task
        except asyncio.CancelledError:
            logger.info("Playback cancelled.")
        finally:
            self._playback_task = None

    async def _play_wav_async(self, wav_data: bytes) -> None:
        """Coroutine that performs blocking PyAudio writes via run_in_executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_wav_blocking, wav_data)

    def _play_wav_blocking(self, wav_data: bytes) -> None:
        """Blocking playback. Prefer PyAudio; fall back to sounddevice."""
        try:
            import pyaudio
        except ImportError:
            self._play_wav_sounddevice_blocking(wav_data)
            return

        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        # Parse WAV header
        try:
            wf = wave.open(io.BytesIO(wav_data), "rb")
        except (wave.Error, EOFError):
            logger.warning("Audio data is not valid WAV; attempting raw playback.")
            self._play_raw_blocking(wav_data)
            return

        with wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()

            format_map = {1: pyaudio.paInt8, 2: pyaudio.paInt16, 4: pyaudio.paInt32}
            pa_format = format_map.get(sample_width, pyaudio.paInt16)

            output_device_index = self._config.tts_output_device_index
            if output_device_index is None:
                output_device_index = (
                    self._pa.get_default_output_device_info()["index"]
                )

            stream = self._pa.open(
                format=pa_format,
                channels=channels,
                rate=framerate,
                output=True,
                output_device_index=output_device_index,
            )

            try:
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                while data and not self._cancel_requested:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
            finally:
                stream.stop_stream()
                stream.close()

    def _play_raw_blocking(self, data: bytes) -> None:
        """Play raw audio bytes as 16-bit mono 24kHz PCM (blocking)."""
        try:
            import pyaudio
        except ImportError:
            self._play_raw_sounddevice_blocking(data)
            return

        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        output_device_index = self._config.tts_output_device_index
        if output_device_index is None:
            output_device_index = (
                self._pa.get_default_output_device_info()["index"]
            )

        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            output_device_index=output_device_index,
        )
        try:
            if not self._cancel_requested:
                stream.write(data)
        finally:
            stream.stop_stream()
            stream.close()

    def _play_wav_sounddevice_blocking(self, wav_data: bytes) -> None:
        """Play WAV bytes via sounddevice when PyAudio is unavailable."""
        try:
            import time
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise TTSError(
                "Playback requires PyAudio or sounddevice+numpy. Install with: "
                "pip install pyaudio  OR  pip install sounddevice numpy"
            ) from exc

        try:
            wf = wave.open(io.BytesIO(wav_data), "rb")
        except (wave.Error, EOFError):
            self._play_raw_sounddevice_blocking(wav_data)
            return

        with wf:
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sample_width != 2:
            raise TTSError(f"sounddevice fallback only supports 16-bit WAV, got {sample_width * 8}-bit")
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=framerate, device=self._config.tts_output_device_index)
        # Poll for cancellation instead of blocking sd.wait() — this lets
        # cancel_playback()'s sd.stop() work reliably without deadlock.
        while not self._cancel_requested:
            try:
                sd.wait()
                break
            except Exception:
                time.sleep(0.02)
        if self._cancel_requested:
            sd.stop()

    def _play_raw_sounddevice_blocking(self, data: bytes) -> None:
        """Play raw 16-bit mono 24 kHz PCM via sounddevice."""
        try:
            import time
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise TTSError(
                "Playback requires PyAudio or sounddevice+numpy. Install with: "
                "pip install pyaudio  OR  pip install sounddevice numpy"
            ) from exc
        audio = np.frombuffer(data, dtype=np.int16)
        sd.play(audio, samplerate=24000, device=self._config.tts_output_device_index)
        if self._cancel_requested:
            sd.stop()
            return
        while not self._cancel_requested:
            try:
                sd.wait()
                break
            except Exception:
                time.sleep(0.02)
        if self._cancel_requested:
            sd.stop()

    # ------------------------------------------------------------------
    # Playback cancellation (interruptible design)
    # ------------------------------------------------------------------

    async def cancel_playback(self) -> None:
        """Cancel any in-progress playback, if running.

        Safe to call at any time — no-op when playback is idle.
        After cancellation, ``_playback_task`` is reset to None.
        """
        self._cancel_requested = True
        try:
            import sounddevice as sd  # type: ignore
            sd.stop()
        except Exception:
            pass
        if self._playback_task is not None and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Clean up PyAudio resources and cancel in-flight playback."""
        await self.cancel_playback()
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
