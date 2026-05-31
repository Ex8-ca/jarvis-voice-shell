"""Edge TTS wrapper for JARVIS Voice Shell.

Uses edge-tts (free Microsoft Edge TTS) to synthesize speech.
Saves to cache_dir as MP3 and plays via PyAudio or sounddevice.

Key design properties:
    - Edge TTS via subprocess CLI (deterministic, no network if cached)
    - Optional edge_tts module (Communicate API) as preferred path
    - MP3 cached in cache_dir with content-hash naming
    - Playback is optional (tts_playback_enabled=False for dry runs)
    - Interruptible/cancellable playback via cancel_playback()
    - **Streaming playback** — audio plays as chunks arrive (~500ms TTFA)
    - No secrets hardcoded — voice/rate come from Config

Known limitations:
    - Requires network for first synthesis (unless cached)
    - No offline fallback in v1
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import io
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Optional

from .config import Config
from .latency import TurnLatency

logger = logging.getLogger(__name__)

# Sentinel value placed in the PCM queue to signal end of synthesis.
_PCM_SENTINEL = object()

# Minimum PCM bytes to accumulate before starting playback.
# This ensures we have enough buffered data to avoid initial underrun.
_MIN_INITIAL_PCM_BYTES = 4800  # ~100ms of 24kHz 16-bit mono


class TTSError(Exception):
    """Raised when TTS synthesis or playback fails."""


class TTSEngine:
    """Synthesize speech and play it through the selected output device.

    Supports both buffered (legacy) and streaming playback modes.
    Streaming mode plays audio chunks as they are synthesized, reducing
    time-to-first-audio from ~3-4s to ~500ms.
    """

    def __init__(self, config: Config):
        self._config = config
        self._pa: object = None  # PyAudio instance (lazy)
        self._output_stream: object = None
        self._playback_task: Optional[asyncio.Task] = None
        self._cancel_requested = False
        """Currently-running playback task, for cancellation."""

        # Streaming-specific state
        self._pcm_queue: Optional[collections.deque] = None
        self._pcm_total = 0
        self._stream_active = False
        self._stream_event: Optional[threading.Event] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._sd_stream: object = None  # sounddevice OutputStream

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

        if latency is not None:
            latency.tts_start = time.perf_counter()

        if self._config.tts_engine == "edge":
            if self._config.tts_streaming_enabled:
                await self._speak_streaming_edge(text, latency=latency)
            else:
                audio_data = await self._synthesize_edge(text)
                if latency is not None:
                    latency.tts_first_audio = time.perf_counter()
                if self._config.tts_playback_enabled:
                    await self._play_wav(audio_data)
        elif self._config.tts_engine == "system":
            audio_data = await self._synthesize_system(text)
            if latency is not None:
                latency.tts_first_audio = time.perf_counter()
            if self._config.tts_playback_enabled:
                await self._play_wav(audio_data)
        else:
            raise TTSError(f"Unknown TTS engine: {self._config.tts_engine}")

        if latency is not None:
            latency.tts_end = time.perf_counter()

    # ------------------------------------------------------------------
    # Streaming TTS (edge-tts)
    # ------------------------------------------------------------------

    async def _speak_streaming_edge(
        self,
        text: str,
        latency: Optional[TurnLatency] = None,
    ) -> None:
        """Stream TTS: play audio as MP3 chunks arrive from edge-tts.

        Flow:
            1. Check cache — if full MP3 exists, use buffered playback (fastest).
            2. Otherwise, start streaming synthesis + playback in parallel.
               - Synthesis yields MP3 chunks via edge-tts Communicate.stream()
               - Each chunk is decoded to PCM via miniaudio
               - PCM is fed to sounddevice/PyAudio immediately
            3. Cache the complete MP3 for future reuse.
        """
        if not self._config.tts_playback_enabled:
            # Still synthesize and cache, but skip playback
            await self._synthesize_edge(text)
            return

        output_path = self._tts_output_path(text)

        # Cache hit: use fast buffered playback (no network needed)
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.debug("TTS cache hit (streaming): %s", output_path)
            mp3_data = output_path.read_bytes()
            wav_data = await self._mp3_to_wav(mp3_data)
            if latency is not None:
                latency.tts_first_audio = time.perf_counter()
            await self._play_wav(wav_data)
            return

        # Cache miss: stream synthesis + playback
        try:
            import edge_tts
        except ImportError:
            # Fall back to non-streaming subprocess synthesis
            logger.debug("edge-tts module not available, using subprocess fallback")
            audio_data = await self._synthesize_edge_subprocess(text)
            if latency is not None:
                latency.tts_first_audio = time.perf_counter()
            await self._play_wav(audio_data)
            return

        # Set up streaming state
        self._cancel_requested = False
        self._pcm_queue = collections.deque()
        self._pcm_total = 0
        self._stream_active = True
        self._stream_event = threading.Event()

        # Collect all MP3 chunks for caching
        all_mp3_chunks: list[bytes] = []

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self._config.tts_voice,
                rate=self._config.tts_rate,
            )

            # Start playback thread — it will block until all PCM is consumed
            playback_future = asyncio.get_event_loop().run_in_executor(
                None, self._playback_thread_worker,
            )

            # Small delay to let playback thread initialize the audio stream
            await asyncio.sleep(0.05)

            first_audio_stamped = False

            # Stream MP3 chunks from edge-tts
            async for chunk in communicate.stream():
                if self._cancel_requested:
                    break

                if chunk["type"] == "audio":
                    mp3_chunk = chunk.get("data", b"")
                    all_mp3_chunks.append(mp3_chunk)

                    # Decode MP3 chunk to PCM using miniaudio
                    try:
                        pcm_data = self._mp3_chunk_to_pcm(mp3_chunk)
                    except Exception as e:
                        logger.warning(
                            "Failed to decode MP3 chunk, falling back to ffmpeg: %s",
                            e,
                        )
                        pcm_data = await self._mp3_chunk_to_pcm_ffmpeg(mp3_chunk)

                    if pcm_data:
                        self._pcm_queue.append(pcm_data)
                        self._pcm_total += len(pcm_data)

                        # Stamp first-audio latency on first PCM chunk
                        if not first_audio_stamped and latency is not None:
                            latency.tts_first_audio = time.perf_counter()
                            first_audio_stamped = True

            # Signal end of synthesis to playback thread
            self._pcm_queue.append(_PCM_SENTINEL)
            self._stream_active = False

        except AttributeError:
            logger.debug(
                "edge-tts Communicate not available, falling back to subprocess",
            )
            audio_data = await self._synthesize_edge_subprocess(text)
            if latency is not None:
                latency.tts_first_audio = time.perf_counter()
            await self._play_wav(audio_data)
            return

        # Wait for playback to finish
        try:
            await playback_future
        except asyncio.CancelledError:
            logger.info("Streaming playback cancelled.")
        except Exception as e:
            if not self._cancel_requested:
                logger.error("Streaming playback error: %s", e)

        # Save complete MP3 to cache
        if all_mp3_chunks and not self._cancel_requested:
            mp3_data = b"".join(all_mp3_chunks)
            try:
                output_path.write_bytes(mp3_data)
                logger.debug("Cached streaming TTS output: %s", output_path)
            except Exception:
                logger.debug("Failed to cache TTS output", exc_info=True)

    def _playback_thread_worker(self) -> None:
        """Blocking thread function that plays PCM data from the queue.

        Opens an audio stream (PyAudio or sounddevice) and feeds it PCM
        data as it arrives in the queue.  Blocks until the sentinel is
        received or cancellation is requested.
        """
        try:
            import pyaudio
            self._playback_thread_worker_pyaudio(pyaudio)
        except ImportError:
            self._playback_thread_worker_sounddevice()

    def _playback_thread_worker_pyaudio(self, pyaudio) -> None:  # type: ignore[override]
        """PyAudio-based streaming playback worker."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        output_device_index = self._config.tts_output_device_index
        if output_device_index is None:
            output_device_index = (
                self._pa.get_default_output_device_info()["index"]  # type: ignore[union-attr]
            )

        stream = self._pa.open(  # type: ignore[union-attr]
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            output_device_index=output_device_index,
        )

        # Wait for initial buffer to accumulate
        initial_timeout = 2.0  # max seconds to wait for first PCM
        start = time.monotonic()
        while (self._pcm_total < _MIN_INITIAL_PCM_BYTES
               and not self._cancel_requested
               and (time.monotonic() - start) < initial_timeout):
            time.sleep(0.01)

        try:
            chunk_size = 2048  # bytes (1024 samples at 16-bit)
            accumulated = bytearray()

            while not self._cancel_requested:
                # Drain accumulated buffer first
                while len(accumulated) >= chunk_size:
                    stream.write(bytes(accumulated[:chunk_size]))
                    del accumulated[:chunk_size]

                # Pull from queue
                if self._pcm_queue:
                    item = self._pcm_queue.popleft()
                    if item is _PCM_SENTINEL:
                        # Flush remaining accumulated data
                        if accumulated:
                            stream.write(bytes(accumulated))
                        return
                    accumulated.extend(item)
                else:
                    time.sleep(0.005)  # brief sleep to avoid busy-wait

        finally:
            stream.stop_stream()
            stream.close()

    def _playback_thread_worker_sounddevice(self) -> None:
        """Sounddevice-based streaming playback worker using callback API."""
        import numpy as np
        import sounddevice as sd

        # Buffer for accumulated PCM data (thread-safe deque of bytes)
        local_buffer: collections.deque = collections.deque()
        local_accumulated = bytearray()
        synthesis_done = threading.Event()
        playback_done = threading.Event()
        stream_holder: list = [None]  # mutable holder for callback closure

        # Minimum samples before we signal ready to start callback
        min_samples = _MIN_INITIAL_PCM_BYTES // 2  # 16-bit samples

        def audio_callback(outdata, frames, time_info, status):
            if status:
                logger.debug("sounddevice callback status: %s", status)

            needed = frames  # number of samples needed
            total_needed = needed  # mono: frames == samples

            # Fill from accumulated
            while len(local_accumulated) < total_needed * 2:  # 2 bytes per sample
                # Try to pull from queue
                if local_buffer:
                    item = local_buffer.popleft()
                    if item is _PCM_SENTINEL:
                        synthesis_done.set()
                        break
                    local_accumulated.extend(item)
                elif synthesis_done.is_set():
                    break
                else:
                    time.sleep(0.002)

            # Extract samples
            available_bytes = min(len(local_accumulated), total_needed * 2)
            sample_bytes = bytes(local_accumulated[:available_bytes])
            del local_accumulated[:available_bytes]

            # Convert to numpy array
            samples = np.frombuffer(sample_bytes, dtype=np.int16)

            # If we got fewer samples than needed, pad with zeros
            if len(samples) < total_needed:
                padding = np.zeros(total_needed - len(samples), dtype=np.int16)
                samples = np.concatenate([samples, padding])

            outdata[:, 0] = samples[:total_needed]

            # Check if we're done
            if synthesis_done.is_set() and not local_accumulated:
                playback_done.set()

        # Wait for initial data to accumulate
        initial_timeout = 2.0
        start = time.monotonic()
        while (self._pcm_total < _MIN_INITIAL_PCM_BYTES
               and not self._cancel_requested
               and (time.monotonic() - start) < initial_timeout):
            time.sleep(0.01)

        # Transfer queue data to local buffer for the callback
        while self._pcm_queue:
            item = self._pcm_queue.popleft()
            local_buffer.append(item)

        # Set up a monitor thread to transfer queue items to local buffer
        def queue_monitor():
            while not self._cancel_requested:
                if self._pcm_queue:
                    item = self._pcm_queue.popleft()
                    local_buffer.append(item)
                    if item is _PCM_SENTINEL:
                        synthesis_done.set()
                        break
                elif self._stream_active:
                    time.sleep(0.005)
                else:
                    # Queue is empty and stream is inactive — check if sentinel came
                    if not synthesis_done.is_set():
                        synthesis_done.set()
                    break

        monitor = threading.Thread(target=queue_monitor, daemon=True)
        monitor.start()

        output_device_index = self._config.tts_output_device_index

        try:
            with sd.OutputStream(
                samplerate=24000,
                channels=1,
                dtype=np.int16,
                callback=audio_callback,
                device=output_device_index,
            ) as sd_stream:
                stream_holder[0] = sd_stream
                self._sd_stream = sd_stream

                # Wait for playback to complete or cancellation
                while not playback_done.is_set() and not self._cancel_requested:
                    playback_done.wait(timeout=0.1)

                if self._cancel_requested:
                    sd_stream.stop()
        except Exception as e:
            if not self._cancel_requested:
                logger.error("sounddevice streaming playback error: %s", e)
        finally:
            self._sd_stream = None
            monitor.join(timeout=1.0)

    @staticmethod
    def _mp3_chunk_to_pcm(mp3_data: bytes) -> bytes:
        """Decode an MP3 chunk to 16-bit 24kHz mono PCM using miniaudio.

        Returns raw PCM bytes (int16, mono, 24kHz).
        """
        import miniaudio

        # Decode MP3 chunk to PCM
        decoded = miniaudio.decode(
            mp3_data,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=24000,
        )

        # decoded.samples is an array.array('h') — convert to bytes
        return decoded.samples.tobytes()

    async def _mp3_chunk_to_pcm_ffmpeg(self, mp3_data: bytes) -> bytes:
        """Fallback: decode MP3 chunk to PCM via ffmpeg subprocess."""
        return await self._mp3_to_wav(mp3_data)

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
                    chunks.append(chunk.get("data", b""))
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
    # Playback (PyAudio / sounddevice)
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

        # Stop any active sounddevice stream (streaming mode)
        if self._sd_stream is not None:
            try:
                import sounddevice as sd
                self._sd_stream.stop()  # type: ignore[union-attr]
                self._sd_stream.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._sd_stream = None

        # Stop any sounddevice playback (legacy mode)
        try:
            import sounddevice as sd  # type: ignore
            sd.stop()
        except Exception:
            pass

        # Wake up the streaming playback thread if waiting on the event
        if self._stream_event is not None:
            self._stream_event.set()

        # Cancel the async playback task (legacy mode)
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
