"""Tests for recorder.py."""

from __future__ import annotations

import wave

import pytest

from jarvis_voice_shell.config import Config
from jarvis_voice_shell.recorder import AudioRecorder, RecorderError
from jarvis_voice_shell.vad import EnergyVAD


class FakeStream:
    def __init__(self):
        self.read_calls = 0
        self.stopped = False
        self.closed = False

    def read(self, chunk_size, exception_on_overflow=False):
        self.read_calls += 1
        return b"\x01\x02" * chunk_size

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakePyAudioInstance:
    def __init__(self):
        self.stream = FakeStream()
        self.open_kwargs = None
        self.terminated = False

    def get_sample_size(self, sample_format):
        return 2

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        return self.stream

    def terminate(self):
        self.terminated = True


class FakePyAudioModule:
    paInt16 = 8

    def __init__(self):
        self.instance = FakePyAudioInstance()

    def PyAudio(self):
        return self.instance


class FakeSoundDeviceModule:
    def __init__(self):
        self.rec_kwargs = None
        self.wait_called = False

    def rec(self, frames, samplerate, channels, dtype, device):
        self.rec_kwargs = {
            "frames": frames,
            "samplerate": samplerate,
            "channels": channels,
            "dtype": dtype,
            "device": device,
        }
        return [[1] for _ in range(frames)]

    def wait(self):
        self.wait_called = True


class FakeNumpyModule:
    int16 = "int16"

    @staticmethod
    def asarray(audio, dtype=None):
        class FakeArray:
            def __init__(self, rows):
                self.rows = rows

            def tobytes(self):
                return b"\x01\x00" * len(self.rows)

        return FakeArray(audio)


def test_record_seconds_writes_wav_and_uses_device(tmp_path):
    fake = FakePyAudioModule()
    cfg = Config(sample_rate=16000, channels=1, chunk_size=800)
    recorder = AudioRecorder(cfg, pyaudio_module=fake)

    result = recorder.record_seconds(0.1, tmp_path / "clip.wav", input_device_index=24)

    assert result.path.exists()
    assert result.device_index == 24
    assert fake.instance.open_kwargs["input_device_index"] == 24
    assert fake.instance.open_kwargs["rate"] == 16000
    assert fake.instance.open_kwargs["channels"] == 1
    assert fake.instance.stream.stopped is True
    assert fake.instance.stream.closed is True

    with wave.open(str(result.path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == result.frames


def test_record_seconds_rejects_zero_duration(tmp_path):
    recorder = AudioRecorder(Config(), pyaudio_module=FakePyAudioModule())
    with pytest.raises(ValueError, match="seconds must be > 0"):
        recorder.record_seconds(0, tmp_path / "clip.wav")


def test_record_seconds_raises_clear_error_when_no_backend(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"pyaudio", "sounddevice", "numpy"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    recorder = AudioRecorder(Config())
    with pytest.raises(RecorderError, match="PyAudio or sounddevice"):
        recorder.record_seconds(0.1, tmp_path / "clip.wav")


def test_record_seconds_falls_back_to_sounddevice(tmp_path):
    sd = FakeSoundDeviceModule()
    cfg = Config(sample_rate=16000, channels=1, chunk_size=800)
    recorder = AudioRecorder(
        cfg,
        pyaudio_module=None,
        sounddevice_module=sd,
        numpy_module=FakeNumpyModule(),
    )

    result = recorder.record_seconds(0.1, tmp_path / "clip.wav", input_device_index=24)

    assert result.backend == "sounddevice"
    assert result.device_index == 24
    assert sd.rec_kwargs["device"] == 24
    assert sd.rec_kwargs["samplerate"] == 16000
    assert sd.wait_called is True
    with wave.open(str(result.path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2


def test_close_terminates_pyaudio_after_record(tmp_path):
    fake = FakePyAudioModule()
    recorder = AudioRecorder(Config(sample_rate=8000, chunk_size=800), pyaudio_module=fake)
    recorder.record_seconds(0.1, tmp_path / "clip.wav")
    recorder.close()
    assert fake.instance.terminated is True


def test_record_until_reads_until_stop_event(tmp_path):
    class StopAfterTwoReads:
        def __init__(self, stream):
            self.stream = stream
        def is_set(self):
            return self.stream.read_calls >= 2

    fake = FakePyAudioModule()
    recorder = AudioRecorder(Config(sample_rate=16000, channels=1, chunk_size=800), pyaudio_module=fake)

    result = recorder.record_until(
        StopAfterTwoReads(fake.instance.stream),
        tmp_path / "ptt.wav",
        input_device_index=24,
        max_seconds=5,
    )

    assert result.path.exists()
    assert result.frames == 1600
    assert result.device_index == 24
    assert result.backend == "pyaudio"
    assert fake.instance.stream.stopped is True
    assert fake.instance.stream.closed is True


def test_record_until_uses_sounddevice_input_stream_and_stop_event(tmp_path):
    class StopAfterThreeReads:
        def __init__(self, stream_ref):
            self.stream_ref = stream_ref
        def is_set(self):
            stream = self.stream_ref[0]
            return stream is not None and stream.read_calls >= 3

    class FakeInputStream:
        def __init__(self, samplerate, channels, dtype, device, blocksize):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.device = device
            self.blocksize = blocksize
            self.read_calls = 0
            stream_ref[0] = self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self, frames):
            self.read_calls += 1
            return ([[1] for _ in range(frames)], False)

    class FakeStreamingSoundDevice:
        InputStream = FakeInputStream

    stream_ref = [None]
    recorder = AudioRecorder(
        Config(sample_rate=16000, channels=1, chunk_size=800),
        pyaudio_module=None,
        sounddevice_module=FakeStreamingSoundDevice,
        numpy_module=FakeNumpyModule(),
    )

    result = recorder.record_until(
        StopAfterThreeReads(stream_ref),
        tmp_path / "ptt-sd.wav",
        input_device_index=24,
        max_seconds=5,
    )

    assert result.backend == "sounddevice-stream"
    assert result.frames == 2400
    assert result.device_index == 24
    assert stream_ref[0].device == 24
    assert stream_ref[0].blocksize == 800
    assert result.path.exists()


def test_record_vad_segment_writes_first_detected_utterance(tmp_path):
    class StopNever:
        def is_set(self):
            return False

    class NumericArray:
        def __init__(self, rows):
            self.rows = rows
        def tobytes(self):
            return b"".join(int(row[0]).to_bytes(2, "little", signed=True) for row in self.rows)

    class NumericNumpy:
        int16 = "int16"
        @staticmethod
        def asarray(audio, dtype=None):
            return NumericArray(audio)

    class FakeInputStream:
        values = [0, 0, 1200, 1300, 1400, 0, 0]
        def __init__(self, samplerate, channels, dtype, device, blocksize):
            self.blocksize = blocksize
            self.read_calls = 0
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self, frames):
            value = self.values[min(self.read_calls, len(self.values) - 1)]
            self.read_calls += 1
            return ([[value] for _ in range(frames)], False)

    class FakeStreamingSoundDevice:
        InputStream = FakeInputStream

    cfg = Config(sample_rate=16000, channels=1, chunk_size=160)
    recorder = AudioRecorder(
        cfg,
        pyaudio_module=None,
        sounddevice_module=FakeStreamingSoundDevice,
        numpy_module=NumericNumpy(),
    )

    result = recorder.record_vad_segment(
        StopNever(),
        tmp_path / "vad.wav",
        input_device_index=24,
        max_seconds=2,
        vad=EnergyVAD(energy_threshold=500, start_frames=2, end_silence_frames=2, pre_roll_frames=1),
    )

    assert result.backend == "sounddevice-vad"
    assert result.frames == 6 * 160
    assert result.path.exists()
    with wave.open(str(result.path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1


def test_record_vad_segment_timeout_reports_peak_rms(tmp_path):
    class StopNever:
        def is_set(self):
            return False

    class NumericArray:
        def __init__(self, rows):
            self.rows = rows
        def tobytes(self):
            return b"".join(int(row[0]).to_bytes(2, "little", signed=True) for row in self.rows)

    class NumericNumpy:
        int16 = "int16"
        @staticmethod
        def asarray(audio, dtype=None):
            return NumericArray(audio)

    class FakeInputStream:
        values = [0, 50, 123, 80]
        def __init__(self, samplerate, channels, dtype, device, blocksize):
            self.read_calls = 0
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self, frames):
            value = self.values[min(self.read_calls, len(self.values) - 1)]
            self.read_calls += 1
            return ([[value] for _ in range(frames)], False)

    class FakeStreamingSoundDevice:
        InputStream = FakeInputStream

    recorder = AudioRecorder(
        Config(sample_rate=16000, channels=1, chunk_size=160),
        pyaudio_module=None,
        sounddevice_module=FakeStreamingSoundDevice,
        numpy_module=NumericNumpy(),
    )

    with pytest.raises(RecorderError, match="peak RMS 123.*threshold 500"):
        recorder.record_vad_segment(
            StopNever(),
            tmp_path / "silent.wav",
            max_seconds=0.04,
            vad=EnergyVAD(energy_threshold=500, start_frames=2, end_silence_frames=2),
        )
