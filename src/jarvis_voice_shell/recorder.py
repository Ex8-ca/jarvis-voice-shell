"""Microphone recording helpers for the JARVIS voice shell.

Phase 1 starts deliberately simple: record a bounded WAV clip from a selected
input device. PyAudio is preferred when installed; sounddevice+numpy is used as
the practical fallback on this Windows box.
"""

from __future__ import annotations

import queue
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .latency import TurnLatency
from .vad import EnergyVAD, VADState, rms_int16


class RecorderError(Exception):
    """Raised when audio capture cannot start or complete."""


@dataclass(frozen=True)
class RecordingResult:
    """Result metadata for a captured WAV file."""

    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    device_index: int | None
    backend: str = "unknown"


class AudioRecorder:
    """Bounded WAV recorder using PyAudio or sounddevice.

    ``pyaudio_module`` and ``sounddevice_module`` are injectable so the recorder
    remains testable without touching real hardware.
    """

    def __init__(
        self,
        config: Config,
        pyaudio_module: Any | None = None,
        sounddevice_module: Any | None = None,
        numpy_module: Any | None = None,
    ):
        self._config = config
        self._pyaudio_module = pyaudio_module
        self._sounddevice_module = sounddevice_module
        self._numpy_module = numpy_module
        self._pa: Any | None = None

    def _load_pyaudio(self) -> Any | None:
        if self._pyaudio_module is not None:
            return self._pyaudio_module
        try:
            import pyaudio  # type: ignore
            return pyaudio
        except ImportError:
            return None

    def _load_sounddevice(self) -> tuple[Any, Any] | None:
        if self._sounddevice_module is not None:
            sd = self._sounddevice_module
        else:
            try:
                import sounddevice as sd  # type: ignore
            except ImportError:
                return None
        if self._numpy_module is not None:
            np = self._numpy_module
        else:
            try:
                import numpy as np  # type: ignore
            except ImportError:
                return None
        return sd, np

    def record_seconds(
        self,
        seconds: float,
        output_path: Path,
        input_device_index: int | None = None,
        latency: TurnLatency | None = None,
    ) -> RecordingResult:
        """Record a fixed-duration WAV clip."""
        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        pyaudio = self._load_pyaudio()
        if pyaudio is not None:
            return self._record_with_pyaudio(seconds, output_path, input_device_index, latency, pyaudio)

        sd_np = self._load_sounddevice()
        if sd_np is not None:
            sd, np = sd_np
            return self._record_with_sounddevice(seconds, output_path, input_device_index, latency, sd, np)

        raise RecorderError(
            "Recording requires PyAudio or sounddevice+numpy. Install with: "
            "pip install pyaudio  OR  pip install sounddevice numpy"
        )

    def record_until(
        self,
        stop_event: Any,
        output_path: Path,
        input_device_index: int | None = None,
        max_seconds: float = 30.0,
        latency: TurnLatency | None = None,
    ) -> RecordingResult:
        """Record until ``stop_event.is_set()`` or ``max_seconds`` elapses.

        This is the PTT capture primitive: key down starts capture, key up sets
        the event, and this method writes a bounded WAV clip. PyAudio supports
        true chunk-by-chunk stop; sounddevice falls back to a bounded recording
        window because its simple ``rec`` API is not streaming.
        """
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")

        pyaudio = self._load_pyaudio()
        if pyaudio is not None:
            return self._record_until_with_pyaudio(
                stop_event, output_path, input_device_index, max_seconds, latency, pyaudio,
            )

        sd_np = self._load_sounddevice()
        if sd_np is not None:
            return self._record_until_with_sounddevice(
                stop_event, output_path, input_device_index, max_seconds, latency, *sd_np,
            )

        raise RecorderError(
            "Recording requires PyAudio or sounddevice+numpy. Install with: "
            "pip install pyaudio  OR  pip install sounddevice numpy"
        )

    def record_vad_segment(
        self,
        stop_event: Any,
        output_path: Path,
        input_device_index: int | None = None,
        max_seconds: float = 30.0,
        latency: TurnLatency | None = None,
        vad: EnergyVAD | None = None,
        on_speech_start: Any | None = None,
    ) -> RecordingResult:
        """Record one complete utterance using always-on energy VAD."""
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        sd_np = self._load_sounddevice()
        if sd_np is None:
            raise RecorderError("Always-on VAD mode requires sounddevice+numpy")
        return self._record_vad_segment_with_sounddevice(
            stop_event, output_path, input_device_index, max_seconds, latency, vad, on_speech_start, *sd_np,
        )

    def _record_until_with_pyaudio(
        self,
        stop_event: Any,
        output_path: Path,
        input_device_index: int | None,
        max_seconds: float,
        latency: TurnLatency | None,
        pyaudio: Any,
    ) -> RecordingResult:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        device_index = input_device_index if input_device_index is not None else self._config.input_device_index
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self._config.sample_rate)
        channels = int(self._config.channels)
        chunk_size = int(self._config.chunk_size)
        sample_format = getattr(pyaudio, "paInt16", 8)
        sample_width = int(self._pa.get_sample_size(sample_format))
        max_frames = max(1, int(sample_rate / chunk_size * max_seconds))

        if latency is not None:
            latency.stt_start = time.perf_counter()

        stream = None
        frames: list[bytes] = []
        try:
            stream = self._pa.open(
                format=sample_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size,
            )
            while len(frames) < max_frames and not stop_event.is_set():
                frames.append(stream.read(chunk_size, exception_on_overflow=False))
        except Exception as exc:  # pragma: no cover - hardware-specific branch
            raise RecorderError(f"Recording failed: {exc}") from exc
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

        if latency is not None:
            latency.stt_end = time.perf_counter()

        self._write_wav(output_path, b"".join(frames), channels, sample_width, sample_rate)
        return RecordingResult(
            path=output_path,
            duration_seconds=(len(frames) * chunk_size) / sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frames=len(frames) * chunk_size,
            device_index=device_index,
            backend="pyaudio",
        )

    def _record_until_with_sounddevice(
        self,
        stop_event: Any,
        output_path: Path,
        input_device_index: int | None,
        max_seconds: float,
        latency: TurnLatency | None,
        sd: Any,
        np: Any,
    ) -> RecordingResult:
        """Record PTT audio with sounddevice's streaming InputStream API."""
        device_index = input_device_index if input_device_index is not None else self._config.input_device_index
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self._config.sample_rate)
        channels = int(self._config.channels)
        chunk_size = int(self._config.chunk_size)
        max_chunks = max(1, int(sample_rate / chunk_size * max_seconds))

        if latency is not None:
            latency.stt_start = time.perf_counter()

        chunks: list[bytes] = []
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device_index,
                blocksize=chunk_size,
            ) as stream:
                while len(chunks) < max_chunks and not stop_event.is_set():
                    audio, _overflowed = stream.read(chunk_size)
                    chunks.append(np.asarray(audio, dtype=np.int16).tobytes())
        except Exception as exc:  # pragma: no cover - hardware-specific branch
            if latency is not None:
                latency.stt_end = time.perf_counter()
            raise RecorderError(f"Recording failed: {exc}") from exc

        if latency is not None:
            latency.stt_end = time.perf_counter()

        data = b"".join(chunks)
        self._write_wav(output_path, data, channels, 2, sample_rate)
        return RecordingResult(
            path=output_path,
            duration_seconds=(len(chunks) * chunk_size) / sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            frames=len(chunks) * chunk_size,
            device_index=device_index,
            backend="sounddevice-stream",
        )

    def _record_vad_segment_with_sounddevice(
        self,
        stop_event: Any,
        output_path: Path,
        input_device_index: int | None,
        max_seconds: float,
        latency: TurnLatency | None,
        vad: EnergyVAD | None,
        on_speech_start: Any | None,
        sd: Any,
        np: Any,
    ) -> RecordingResult:
        """Record one VAD-delimited utterance with sounddevice InputStream."""
        device_index = input_device_index if input_device_index is not None else self._config.input_device_index
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self._config.sample_rate)
        channels = int(self._config.channels)
        chunk_size = int(self._config.chunk_size)
        max_chunks = max(1, int(sample_rate / chunk_size * max_seconds))
        if vad is None:
            end_frames = max(1, int((self._config.vad_end_silence_ms / 1000) * sample_rate / chunk_size))
            pre_roll_frames = max(0, int((self._config.vad_pre_roll_ms / 1000) * sample_rate / chunk_size))
            vad = EnergyVAD(
                energy_threshold=self._config.vad_energy_threshold,
                start_frames=self._config.vad_start_frames,
                end_silence_frames=end_frames,
                pre_roll_frames=pre_roll_frames,
            )

        if latency is not None:
            latency.stt_start = time.perf_counter()

        segment: bytes | None = None
        chunks_read = 0
        peak_rms = 0
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device_index,
                blocksize=chunk_size,
            ) as stream:
                while chunks_read < max_chunks and not stop_event.is_set():
                    audio, _overflowed = stream.read(chunk_size)
                    chunks_read += 1
                    frame = np.asarray(audio, dtype=np.int16).tobytes()
                    peak_rms = max(peak_rms, rms_int16(frame))
                    prior_state = vad.state
                    segment = vad.process(frame)
                    if prior_state != VADState.SPEAKING and vad.state == VADState.SPEAKING and on_speech_start is not None:
                        on_speech_start()
                    if segment is not None:
                        break
        except Exception as exc:  # pragma: no cover - hardware-specific branch
            if "Blocking API not supported" in str(exc) or "Invalid device" in str(exc) or "Insufficient memory" in str(exc):
                return self._record_vad_segment_with_sounddevice_raw_callback(
                    stop_event, output_path, device_index, max_seconds, latency, vad, on_speech_start, sd, sample_rate, channels, chunk_size
                )
            if latency is not None:
                latency.stt_end = time.perf_counter()
            raise RecorderError(f"VAD recording failed: {exc}") from exc

        if latency is not None:
            latency.stt_end = time.perf_counter()
        if segment is None:
            raise RecorderError(f"No speech detected before timeout; peak RMS {peak_rms}, threshold {vad.energy_threshold}")

        self._write_wav(output_path, segment, channels, 2, sample_rate)
        return RecordingResult(
            path=output_path,
            duration_seconds=(len(segment) // (2 * channels)) / sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            frames=len(segment) // (2 * channels),
            device_index=device_index,
            backend="sounddevice-vad",
        )

    def _record_vad_segment_with_sounddevice_raw_callback(
        self,
        stop_event: Any,
        output_path: Path,
        device_index: int | None,
        max_seconds: float,
        latency: TurnLatency | None,
        vad: EnergyVAD,
        on_speech_start: Any | None,
        sd: Any,
        sample_rate: int,
        channels: int,
        chunk_size: int,
    ) -> RecordingResult:
        """Record VAD via RawInputStream callback for WDM-KS devices.

        Some Windows headphone/headset routes do not support PortAudio's blocking
        read API. Raw callback mode works for those devices, notably the Oculus
        virtual/headphone microphone route on this machine.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        max_chunks = max(1, int(sample_rate / chunk_size * max_seconds))
        q: queue.Queue[bytes] = queue.Queue()

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            q.put(bytes(indata))

        if latency is not None and latency.stt_start is None:
            latency.stt_start = time.perf_counter()

        segment: bytes | None = None
        chunks_read = 0
        peak_rms = 0
        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device_index,
                blocksize=chunk_size,
                callback=_callback,
            ):
                while chunks_read < max_chunks and not stop_event.is_set():
                    try:
                        frame = q.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    chunks_read += 1
                    peak_rms = max(peak_rms, rms_int16(frame))
                    prior_state = vad.state
                    segment = vad.process(frame)
                    if prior_state != VADState.SPEAKING and vad.state == VADState.SPEAKING and on_speech_start is not None:
                        on_speech_start()
                    if segment is not None:
                        break
        except Exception as exc:  # pragma: no cover - hardware-specific branch
            if latency is not None:
                latency.stt_end = time.perf_counter()
            raise RecorderError(f"VAD recording failed: {exc}") from exc

        if latency is not None:
            latency.stt_end = time.perf_counter()
        if segment is None:
            raise RecorderError(f"No speech detected before timeout; peak RMS {peak_rms}, threshold {vad.energy_threshold}")

        self._write_wav(output_path, segment, channels, 2, sample_rate)
        return RecordingResult(
            path=output_path,
            duration_seconds=(len(segment) // (2 * channels)) / sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            frames=len(segment) // (2 * channels),
            device_index=device_index,
            backend="sounddevice-vad-raw-callback",
        )

    def _record_with_pyaudio(
        self,
        seconds: float,
        output_path: Path,
        input_device_index: int | None,
        latency: TurnLatency | None,
        pyaudio: Any,
    ) -> RecordingResult:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        device_index = input_device_index if input_device_index is not None else self._config.input_device_index
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = int(self._config.sample_rate)
        channels = int(self._config.channels)
        chunk_size = int(self._config.chunk_size)
        sample_format = getattr(pyaudio, "paInt16", 8)
        sample_width = int(self._pa.get_sample_size(sample_format))
        frame_target = max(1, int(sample_rate / chunk_size * seconds))

        if latency is not None:
            latency.stt_start = time.perf_counter()

        stream = None
        frames: list[bytes] = []
        try:
            stream = self._pa.open(
                format=sample_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size,
            )
            for _ in range(frame_target):
                frames.append(stream.read(chunk_size, exception_on_overflow=False))
        except Exception as exc:  # pragma: no cover - hardware-specific branch
            raise RecorderError(f"Recording failed: {exc}") from exc
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

        if latency is not None:
            latency.stt_end = time.perf_counter()

        self._write_wav(output_path, b"".join(frames), channels, sample_width, sample_rate)
        return RecordingResult(
            path=output_path,
            duration_seconds=seconds,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frames=len(frames) * chunk_size,
            device_index=device_index,
            backend="pyaudio",
        )

    def _record_with_sounddevice(
        self,
        seconds: float,
        output_path: Path,
        input_device_index: int | None,
        latency: TurnLatency | None,
        sd: Any,
        np: Any,
    ) -> RecordingResult:
        device_index = input_device_index if input_device_index is not None else self._config.input_device_index
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self._config.sample_rate)
        channels = int(self._config.channels)
        frames = max(1, int(seconds * sample_rate))

        if latency is not None:
            latency.stt_start = time.perf_counter()

        try:
            audio = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device_index,
            )
            sd.wait()
        except Exception as exc:  # pragma: no cover - hardware-specific branch
            raise RecorderError(f"Recording failed: {exc}") from exc

        if latency is not None:
            latency.stt_end = time.perf_counter()

        data = np.asarray(audio, dtype=np.int16).tobytes()
        self._write_wav(output_path, data, channels, 2, sample_rate)
        return RecordingResult(
            path=output_path,
            duration_seconds=seconds,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            frames=frames,
            device_index=device_index,
            backend="sounddevice",
        )

    @staticmethod
    def _write_wav(path: Path, data: bytes, channels: int, sample_width: int, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(data)

    def close(self) -> None:
        """Release PyAudio if it was opened."""
        if self._pa is not None:
            try:
                self._pa.terminate()
            finally:
                self._pa = None
