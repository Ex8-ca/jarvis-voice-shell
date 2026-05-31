"""Tests for tts.py — edge-tts command builder, file synthesis, playback."""

import asyncio
import io
import sys
import wave
from pathlib import Path

import pytest

from jarvis_voice_shell.config import Config
from jarvis_voice_shell.tts import TTSEngine, TTSError


# ---------------------------------------------------------------------------
# Helper: mock edge-tts subprocess
# ---------------------------------------------------------------------------

class _FakeSubprocess:
    """Simulates a completed subprocess with given returncode and output."""

    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self, input=None):
        return self._stdout, self._stderr


def _make_mock_exec(returncode=0, stderr=b"", on_call=None):
    """Return a mock for asyncio.create_subprocess_exec.

    on_call: optional callback(args, kwargs) called before returning.
    """
    async def mock_exec(*args, **kwargs):
        if on_call:
            on_call(args, kwargs)
        return _FakeSubprocess(returncode=returncode, stderr=stderr)
    return mock_exec


# ---------------------------------------------------------------------------
# Existing tests preserved
# ---------------------------------------------------------------------------

class TestTTSEngineInit:
    """TTS engine initialization."""

    def test_creates_with_config(self):
        cfg = Config()
        tts = TTSEngine(cfg)
        assert tts._config is cfg

    def test_default_engine_is_edge(self):
        tts = TTSEngine(Config())
        assert tts._config.tts_engine == "edge"


class TestTTSError:
    """TTSError is a standard exception."""

    def test_is_exception(self):
        with pytest.raises(TTSError):
            raise TTSError("synthesis failed")

    def test_str_representation(self):
        err = TTSError("invalid voice")
        assert str(err) == "invalid voice"


class TestSpeakEdgeCases:
    """Edge cases for speak()."""

    @pytest.mark.asyncio
    async def test_empty_text_is_noop(self):
        tts = TTSEngine(Config())
        await tts.speak("")
        await tts.speak("   ")

    @pytest.mark.asyncio
    async def test_unknown_engine_raises(self):
        cfg = Config(tts_engine="nonexistent")
        tts = TTSEngine(cfg)
        with pytest.raises(TTSError, match="Unknown TTS engine"):
            await tts.speak("hello")

    @pytest.mark.asyncio
    async def test_system_tts_not_implemented(self):
        cfg = Config(tts_engine="system")
        tts = TTSEngine(cfg)
        with pytest.raises(TTSError, match="System TTS not implemented"):
            await tts.speak("hello")


class TestTTsClose:
    """close() cleanup."""

    @pytest.mark.asyncio
    async def test_close_no_pa_is_noop(self):
        tts = TTSEngine(Config())
        await tts.close()  # should not raise


# ---------------------------------------------------------------------------
# Edge TTS command construction
# ---------------------------------------------------------------------------

class TestEdgeTtsCommand:
    """Construction of edge-tts CLI command from config."""

    def test_default_command(self, tmp_path):
        cfg = Config(tts_voice="en-GB-RyanNeural", tts_rate="+0%")
        tts = TTSEngine(cfg)
        out_path = tmp_path / "output.mp3"
        cmd = tts._construct_edge_tts_command("Hello world", out_path)
        assert cmd[0] == "edge-tts"
        assert "--voice" in cmd
        assert "en-GB-RyanNeural" in cmd
        assert "--rate" in cmd
        assert "+0%" in cmd
        assert "--text" in cmd
        assert "Hello world" in cmd
        assert "--write-media" in cmd
        assert str(out_path) in cmd

    def test_command_custom_voice(self, tmp_path):
        cfg = Config(tts_voice="en-US-AriaNeural", tts_rate="-5%")
        tts = TTSEngine(cfg)
        cmd = tts._construct_edge_tts_command("Test", tmp_path / "out.mp3")
        assert "en-US-AriaNeural" in cmd
        assert "-5%" in cmd

    def test_command_handles_special_characters(self, tmp_path):
        cfg = Config()
        tts = TTSEngine(cfg)
        out_path = tmp_path / "out.mp3"
        cmd = tts._construct_edge_tts_command(
            'He said "hello" — test', out_path,
        )
        assert "He said" in " ".join(cmd)

    def test_command_result_is_list_of_strings(self, tmp_path):
        cfg = Config()
        tts = TTSEngine(cfg)
        cmd = tts._construct_edge_tts_command("text", tmp_path / "out.mp3")
        assert isinstance(cmd, list)
        assert all(isinstance(a, str) for a in cmd)


# ---------------------------------------------------------------------------
# File-based synthesis
# ---------------------------------------------------------------------------

class TestSynthesizeToFile:
    """synthesize_edge_to_file — MP3 written to cache_dir."""

    @pytest.mark.asyncio
    async def test_returns_path_on_success(self, tmp_path, monkeypatch):
        cfg = Config(cache_dir=tmp_path)
        tts = TTSEngine(cfg)

        # Create the expected output file path
        out_path = tts._tts_output_path("hello world")
        out_path.write_text("fake mp3 data")

        mock_exec = _make_mock_exec(returncode=0)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

        result = await tts.synthesize_edge_to_file("hello world")
        assert isinstance(result, Path)
        # File should exist (written by mock or already there)
        assert result == out_path

    @pytest.mark.asyncio
    async def test_uses_cache_dir(self, tmp_path, monkeypatch):
        cfg = Config(cache_dir=tmp_path)
        tts = TTSEngine(cfg)

        captured_path = []

        def on_call(args, kwargs):
            for a in args:
                s = str(a)
                if s.endswith(".mp3"):
                    captured_path.append(s)
                    Path(s).write_text("fake mp3")

        mock_exec = _make_mock_exec(returncode=0, on_call=on_call)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

        result = await tts.synthesize_edge_to_file("test text")
        assert result.parent == tmp_path

    @pytest.mark.asyncio
    async def test_raises_on_subprocess_failure(self, tmp_path, monkeypatch):
        cfg = Config(cache_dir=tmp_path)
        tts = TTSEngine(cfg)

        mock_exec = _make_mock_exec(
            returncode=1, stderr=b"edge-tts: synthesis failed",
        )
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

        with pytest.raises(TTSError, match="edge-tts"):
            await tts.synthesize_edge_to_file("fail")

    @pytest.mark.asyncio
    async def test_output_filename_is_deterministic(self, tmp_path, monkeypatch):
        cfg = Config(cache_dir=tmp_path)
        tts = TTSEngine(cfg)

        def on_call(args, kwargs):
            for a in args:
                s = str(a)
                if s.endswith(".mp3") and not Path(s).exists():
                    Path(s).write_text("fake mp3")

        mock_exec = _make_mock_exec(returncode=0, on_call=on_call)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

        p1 = await tts.synthesize_edge_to_file("hello world")
        p2 = await tts.synthesize_edge_to_file("hello world")
        assert p1 == p2


# ---------------------------------------------------------------------------
# Playback control
# ---------------------------------------------------------------------------

class TestPlaybackControl:
    """Interruptible/cancellable playback design."""

    def test_playback_enabled_default(self):
        cfg = Config()
        assert cfg.tts_playback_enabled is True

    def test_playback_disabled_config(self):
        cfg = Config(tts_playback_enabled=False)
        assert cfg.tts_playback_enabled is False

    @pytest.mark.asyncio
    async def test_speak_skips_playback_when_disabled(self, monkeypatch, tmp_path):
        cfg = Config(tts_playback_enabled=False, cache_dir=tmp_path)
        tts = TTSEngine(cfg)

        synthesis_called = False
        playback_called = False

        orig_synth = tts._synthesize_edge

        async def mock_synth(text):
            nonlocal synthesis_called
            synthesis_called = True
            return b"fake wav"

        tts._synthesize_edge = mock_synth

        async def mock_play(data):
            nonlocal playback_called
            playback_called = True

        tts._play_wav = mock_play

        await tts.speak("test")

        assert synthesis_called, "Synthesis should still happen"
        assert not playback_called, "Playback should be skipped when disabled"

        tts._synthesize_edge = orig_synth

    def test_playback_task_is_stored(self):
        tts = TTSEngine(Config())
        assert hasattr(tts, "_playback_task")
        assert tts._playback_task is None

    @pytest.mark.asyncio
    async def test_cancel_playback_is_noop_when_idle(self):
        tts = TTSEngine(Config())
        await tts.cancel_playback()

    @pytest.mark.asyncio
    async def test_cancel_playback_cancels_task(self):
        tts = TTSEngine(Config())

        async def dummy_play():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        tts._playback_task = asyncio.ensure_future(dummy_play())
        assert not tts._playback_task.done()

        await tts.cancel_playback()
        assert tts._playback_task is None

    @pytest.mark.asyncio
    async def test_close_cancels_playback(self):
        tts = TTSEngine(Config())

        async def dummy_play():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        tts._playback_task = asyncio.ensure_future(dummy_play())
        await tts.close()
        assert tts._playback_task is None

    def test_sounddevice_wav_playback_fallback(self, monkeypatch):
        class FakeAudio:
            def __init__(self, data):
                self.data = data
                self.reshaped = None

            def reshape(self, *shape):
                self.reshaped = shape
                return self

        class FakeNumpy:
            int16 = "int16"

            @staticmethod
            def frombuffer(data, dtype):
                return FakeAudio(data)

        class FakeSoundDevice:
            played = None
            waited = False

            @staticmethod
            def play(audio, samplerate, device):
                FakeSoundDevice.played = (audio, samplerate, device)

            @staticmethod
            def wait():
                FakeSoundDevice.waited = True

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x01\x00" * 10)

        monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)
        monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
        tts = TTSEngine(Config(tts_output_device_index=35))
        tts._play_wav_sounddevice_blocking(wav_buf.getvalue())

        assert FakeSoundDevice.played[1] == 24000
        assert FakeSoundDevice.played[2] == 35
        assert FakeSoundDevice.waited is True


# ---------------------------------------------------------------------------
# file_path helper
# ---------------------------------------------------------------------------

class TestTtsFilePath:
    """Output file path construction."""

    def test_deterministic_path(self):
        cfg = Config(tts_voice="en-GB-RyanNeural", tts_rate="+0%")
        tts = TTSEngine(cfg)
        p1 = tts._tts_output_path("hello world")
        p2 = tts._tts_output_path("hello world")
        assert p1 == p2

    def test_different_text_different_path(self):
        cfg = Config()
        tts = TTSEngine(cfg)
        p1 = tts._tts_output_path("hello")
        p2 = tts._tts_output_path("goodbye")
        assert p1 != p2

    def test_different_voice_different_path(self):
        cfg1 = Config(tts_voice="en-GB-RyanNeural")
        cfg2 = Config(tts_voice="en-US-AriaNeural")
        tts1 = TTSEngine(cfg1)
        tts2 = TTSEngine(cfg2)
        p1 = tts1._tts_output_path("hello")
        p2 = tts2._tts_output_path("hello")
        assert p1 != p2

    def test_path_has_mp3_extension(self):
        cfg = Config()
        tts = TTSEngine(cfg)
        p = tts._tts_output_path("hello")
        assert p.suffix == ".mp3"

    def test_path_is_under_cache_dir(self):
        cfg = Config()
        tts = TTSEngine(cfg)
        p = tts._tts_output_path("hello")
        assert p.parent == cfg.cache_dir


@pytest.mark.asyncio
async def test_cancel_playback_requests_sounddevice_stop(monkeypatch):
    class FakeSoundDevice:
        stopped = False

        @staticmethod
        def stop():
            FakeSoundDevice.stopped = True

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    tts = TTSEngine(Config())

    await tts.cancel_playback()

    assert tts._cancel_requested is True
    assert FakeSoundDevice.stopped is True


# ---------------------------------------------------------------------------
# Streaming TTS tests
# ---------------------------------------------------------------------------


class TestStreamingConfig:
    """Configuration for streaming TTS."""

    def test_streaming_enabled_default(self):
        cfg = Config()
        assert cfg.tts_streaming_enabled is True

    def test_streaming_can_be_disabled(self):
        cfg = Config(tts_streaming_enabled=False)
        assert cfg.tts_streaming_enabled is False


class TestMp3ChunkToPcm:
    """MP3 chunk to PCM conversion via miniaudio."""

    def test_mp3_chunk_to_pcm_static_method_exists(self):
        assert hasattr(TTSEngine, "_mp3_chunk_to_pcm")
        # Verify it's callable (static method)
        # We can't test with real MP3 without network, but verify the method exists

    def test_mp3_chunk_to_pcm_rejects_invalid_data(self):
        """Invalid MP3 data should raise an exception."""
        with pytest.raises(Exception):  # miniaudio.DecodeError or similar
            TTSEngine._mp3_chunk_to_pcm(b"not valid mp3 data at all")


class TestStreamingState:
    """Streaming-specific state management."""

    def test_streaming_state_initialized(self):
        tts = TTSEngine(Config())
        assert tts._pcm_queue is None
        assert tts._pcm_total == 0
        assert tts._stream_active is False
        assert tts._sd_stream is None

    def test_cancel_playback_resets_streaming_state(self):
        """cancel_playback() should stop any active sounddevice stream."""
        import threading

        class FakeStream:
            stopped = False
            closed = False

            def stop(self):
                FakeStream.stopped = True

            def close(self):
                FakeStream.closed = True

        class FakeSoundDevice:
            @staticmethod
            def stop():
                pass

        import sys

        monkeypatch = type("FakeMonkeypatch", (), {"setitem": lambda s, m, v: None})()
        sys.modules["sounddevice"] = FakeSoundDevice

        tts = TTSEngine(Config())
        tts._sd_stream = FakeStream()
        tts._stream_event = threading.Event()

        # Run cancel in an event loop
        import asyncio

        asyncio.get_event_loop().run_until_complete(tts.cancel_playback())

        assert FakeStream.stopped is True
        assert FakeStream.closed is True
        assert tts._sd_stream is None


class TestSpeakWithStreamingDisabled:
    """Verify speak() falls back to buffered mode when streaming is disabled."""

    @pytest.mark.asyncio
    async def test_speak_uses_buffered_mode_when_streaming_disabled(self, monkeypatch, tmp_path):
        cfg = Config(
            cache_dir=tmp_path,
            tts_streaming_enabled=False,
            tts_playback_enabled=False,
        )
        tts = TTSEngine(cfg)

        synthesis_called = False

        async def mock_synth(text):
            nonlocal synthesis_called
            synthesis_called = True
            return b"fake wav"

        tts._synthesize_edge = mock_synth

        await tts.speak("test text with streaming disabled")

        assert synthesis_called, "Should use _synthesize_edge (buffered mode)"


class TestSpeakWithStreamingEnabled:
    """Verify speak() uses streaming mode when enabled."""

    @pytest.mark.asyncio
    async def test_speak_uses_streaming_when_enabled_and_cache_miss(self, monkeypatch, tmp_path):
        cfg = Config(
            cache_dir=tmp_path,
            tts_streaming_enabled=True,
            tts_playback_enabled=False,
        )
        tts = TTSEngine(cfg)

        streaming_called = False

        async def mock_streaming(text, latency=None):
            nonlocal streaming_called
            streaming_called = True

        tts._speak_streaming_edge = mock_streaming

        await tts.speak("test text with streaming enabled")

        assert streaming_called, "Should use _speak_streaming_edge"

    @pytest.mark.asyncio
    async def test_speak_uses_buffered_on_cache_hit(self, monkeypatch, tmp_path):
        """When cache hit, streaming mode should still use buffered playback (faster)."""
        cfg = Config(
            cache_dir=tmp_path,
            tts_streaming_enabled=True,
            tts_playback_enabled=False,
        )
        tts = TTSEngine(cfg)

        # Pre-populate cache
        output_path = tts._tts_output_path("cached text")
        output_path.write_bytes(b"fake mp3 data for cache")

        buffered_synth_called = False

        async def mock_synth(text):
            nonlocal buffered_synth_called
            buffered_synth_called = True
            return b"fake wav"

        tts._synthesize_edge = mock_synth

        await tts.speak("cached text")

        # On cache hit, should use buffered synthesis path
        assert buffered_synth_called, "Should use buffered synthesis on cache hit"
