"""
JARVIS Voice Client — Python client for the JARVIS WebSocket gateway.

- Captures mic from default audio device
- Sends PCM frames over WebSocket to the JARVIS gateway
- Receives TTS MP3 chunks and plays through default audio device
- Barge-in: detects user voice during TTS playback and interrupts the AI
  (requires gateway with barge-in support, e.g. web/jarvis_web.py v2.4+)

Use this for split-architecture deployments: mic on one machine, the
JARVIS gateway (which runs Whisper STT, LLM, and TTS) on another.

For single-machine deployments, use the web UI at web/jarvis_web.py instead.

Usage:
    python3 jarvis_voice_client.py [--host <gateway-host>] [--port <gateway-port>]

Or via environment:
    JARVIS_WS_HOST=192.168.1.50 JARVIS_WS_PORT=8989 python3 jarvis_voice_client.py

Dependencies (in venv):
    sounddevice, numpy, websockets, miniaudio
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import queue
import signal
import sys
import threading
from typing import Optional

import sounddevice as sd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis-voice-client")


# ── Config ────────────────────────────────────────────────────────────

# Defaults to localhost — set JARVIS_WS_HOST to your gateway's IP for split setups.
# Default port 8989 (jarvis_web.py) supports barge-in. Port 6790 is the older
# jarvis_ws_gateway.py which doesn't support barge-in (graceful degradation).
WS_HOST = os.environ.get("JARVIS_WS_HOST", "127.0.0.1")
WS_PORT = int(os.environ.get("JARVIS_WS_PORT", "8989"))
WS_URL = f"ws://{WS_HOST}:{WS_PORT}/ws"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE * 63 // 1000  # 1008
BYTES_PER_FRAME = SAMPLES_PER_FRAME * AUDIO_SAMPLE_WIDTH  # 2016

# ── Barge-in tuning ───────────────────────────────────────────────────
# SIDETONE_DELAY_MS: estimated time between TTS audio leaving the speakers
# and being picked up by the mic. Subtract this much TTS ref from the
# mic frame. Tune for your hardware.
SIDETONE_DELAY_MS = int(os.environ.get("JARVIS_SIDETONE_DELAY_MS", "80"))
SIDETONE_DELAY_SAMPLES = AUDIO_SAMPLE_RATE * SIDETONE_DELAY_MS // 1000  # e.g. 1280

# BARGE_IN_RMS: minimum RMS amplitude on the CLEANED mic to trigger
# barge-in while TTS is playing. Higher = more deliberate speech required
# to interrupt. Recommended: 600-1500.
BARGE_IN_RMS = int(os.environ.get("JARVIS_BARGE_IN_RMS", "800"))

# Local VAD energy threshold for normal speech (when TTS is not playing).
# Server uses the same threshold (300) but our local VAD has the *cleaned*
# mic which is more sensitive after sidetone cancellation.
LOCAL_VAD_RMS = int(os.environ.get("JARVIS_LOCAL_VAD_RMS", "400"))


# ── Device selection ──────────────────────────────────────────────────

def get_devices() -> tuple[dict, dict]:
    devs = sd.query_devices()
    inputs, outputs = {}, {}
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            inputs[i] = d
        if d.get("max_output_channels", 0) > 0:
            outputs[i] = d
    return inputs, outputs


# ── PCM Capture (background thread → queue) ─────────────────────────

class MicCapture:
    """Capture mic via sounddevice, put PCM frames in a thread-safe queue.

    Optional sidetone cancellation: if a `tts_ref_buffer` is attached, the
    mic frame is subtracted from the TTS reference before being queued.
    The subtraction removes the AI's own voice bleeding from speakers→mic
    so the server's VAD doesn't trigger on the AI's audio.
    """

    def __init__(self, device: Optional[int] = None, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.device = device
        self.sample_rate = sample_rate
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Optional sidetone cancellation: TTSRefBuffer instance (or None)
        self._tts_ref = None
        # State exposed for the main loop to read:
        # - last_clean_rms: RMS of the last *cleaned* mic frame
        # - is_sidetoning: True if TTS ref was subtracted on the last frame
        self.last_clean_rms: float = 0.0
        self.is_sidetoning: bool = False

    def attach_tts_ref(self, tts_ref_buffer) -> None:
        """Attach a TTS reference buffer for sidetone cancellation."""
        self._tts_ref = tts_ref_buffer

    def start(self) -> None:
        self._running = True
        self._stream = sd.InputStream(
            device=self.device,
            channels=AUDIO_CHANNELS,
            samplerate=self.sample_rate,
            blocksize=SAMPLES_PER_FRAME,
            dtype="int16",
            callback=self._on_frame,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        dev = sd.query_devices(self.device if self.device is not None else None)
        logger.info(
            "Mic capture: device=%s, rate=%.0f, frames=%d bytes",
            dev.get("name", "default") if self.device is not None else "default (PipeWire)",
            dev.get("default_samplerate", self.sample_rate),
            BYTES_PER_FRAME,
        )

    def _on_frame(self, indata: np.ndarray, frames: int, status: sd.CallbackFlags, _):
        if status:
            logger.debug("capture status: %s", status)
        try:
            # indata shape: (frames, channels) — mono so [:, 0]
            mic = indata[:, 0].astype(np.int32)  # int32 for arithmetic headroom
            self.is_sidetoning = False

            # ── Sidetone cancellation ──────────────────────────────────
            # If TTS is playing, subtract the corresponding window of the
            # AI's audio from the mic input. This removes the AI's voice
            # bleeding through the speakers→mic path so VAD sees only the
            # user's voice.
            if self._tts_ref is not None and self._tts_ref.is_active:
                ref_samples = self._tts_ref.read(len(mic))
                if ref_samples is not None and len(ref_samples) == len(mic):
                    mic = mic - ref_samples
                    self.is_sidetoning = True

            # Clip back to int16 range
            np.clip(mic, -32768, 32767, out=mic)
            mic_int16 = mic.astype("<h")
            pcm = bytes(mic_int16.tobytes())

            # Compute RMS for barge-in detection (on cleaned signal)
            self.last_clean_rms = float(np.sqrt(np.mean(mic_int16.astype(np.float32) ** 2)))

            self._queue.put_nowait(pcm)
        except queue.Full:
            pass

    def _run(self):
        try:
            with self._stream:
                while self._running:
                    sd.sleep(100)
        except Exception as e:
            logger.error("Mic capture error: %s", e)

    def read(self) -> Optional[bytes]:
        """Non-blocking read — returns None if no frame ready."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._stream = None


# ── TTS Reference Buffer (sidetone cancellation) ────────────────────

class TTSRefBuffer:
    """Thread-safe ring buffer of TTS PCM samples for sidetone cancellation.

    `play()` adds decoded PCM samples to the buffer.
    `read(n)` pops samples from the front (FIFO) and returns them.

    Used by MicCapture to subtract the AI's voice from the mic input.
    The buffer should hold at least SIDETONE_DELAY_SAMPLES + a few mic
    frames worth of audio (~3-4KB at 16kHz) to handle the playback delay.
    """

    def __init__(self, maxlen_samples: int = 32000):
        self._buf: collections.deque = collections.deque(maxlen=maxlen_samples)
        self._lock = threading.Lock()
        self._is_active = False
        self._stopped_at = 0.0  # monotonic time when TTS was last active
        import time as _t
        self._time = _t

    def append(self, pcm_int16: np.ndarray) -> None:
        """Add a chunk of int16 PCM samples to the buffer."""
        with self._lock:
            self._buf.extend(pcm_int16.tolist())
            self._is_active = True
            self._stopped_at = self._time.monotonic()

    def read(self, n: int) -> Optional[np.ndarray]:
        """Pop and return up to n samples from the front of the buffer.

        If the buffer has fewer than n samples and TTS is still considered
        active (within 200ms of last append), pad with zeros.
        Returns None if TTS is not active and the buffer is empty.
        """
        with self._lock:
            if not self._buf:
                if self._is_active and (self._time.monotonic() - self._stopped_at) < 0.2:
                    # TTS just ended but the mic might still be picking up audio
                    # → return zeros so subtraction has a defined value
                    return np.zeros(n, dtype=np.int32)
                self._is_active = False
                return None
            samples = [self._buf.popleft() for _ in range(min(n, len(self._buf)))]
            if len(samples) < n:
                # Pad with zeros if we ran out mid-frame
                samples.extend([0] * (n - len(samples)))
            return np.array(samples, dtype=np.int32)

    @property
    def is_active(self) -> bool:
        """True if TTS has been playing recently (within 200ms)."""
        if self._is_active and (self._time.monotonic() - self._stopped_at) > 0.2:
            self._is_active = False
        return self._is_active

    def clear(self) -> None:
        """Drop all buffered TTS samples (used on barge-in)."""
        with self._lock:
            self._buf.clear()
            self._is_active = False


# ── MP3 Decoder via miniaudio ────────────────────────────────────────

def decode_mp3(mp3_data: bytes) -> bytes:
    """Decode MP3 bytes → 16-bit mono PCM at 16kHz using miniaudio."""
    import miniaudio as ma

    try:
        decoded = ma.decode(
            mp3_data,
            output_format=ma.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=AUDIO_SAMPLE_RATE,
        )
        samples = decoded.samples
        if samples is None or len(samples) == 0:
            return b""

        # Convert array.array to bytes
        if hasattr(samples, 'tobytes'):
            return bytes(samples.tobytes())
        return bytes(samples)
    except Exception as e:
        logger.warning("MP3 decode failed: %s", e)
        return b""


# ── Audio playback (async queue → sounddevice) ────────────────────────

class Speaker:
    """Play PCM chunks via sounddevice output stream.

    If a `tts_ref_buffer` is attached, decoded TTS PCM is *also* pushed
    into that buffer for sidetone cancellation in MicCapture.
    """

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._stream: Optional[sd.OutputStream] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Optional TTS reference buffer (sidetone cancellation)
        self._tts_ref = None
        # Set True while TTS audio is on the queue or playing
        self._is_playing_tts = False
        self._playback_lock = threading.Lock()

    def attach_tts_ref(self, tts_ref_buffer) -> None:
        """Attach a TTS reference buffer for sidetone cancellation."""
        self._tts_ref = tts_ref_buffer

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._play_loop())
        dev = sd.query_devices(self.device if self.device is not None else None)
        logger.info(
            "Speaker: device=%s, rate=%d",
            dev.get("name", "default") if self.device is not None else "default (PipeWire)",
            AUDIO_SAMPLE_RATE,
        )

    async def _play_loop(self) -> None:
        def callback(outdata: np.ndarray, frames: int, status: sd.CallbackFlags, _):
            if status:
                logger.debug("speaker status: %s", status)
            try:
                chunk = self._queue.get_nowait()
                arr = np.frombuffer(chunk, dtype="<h")
                if len(arr) < frames:
                    arr = np.pad(arr, (0, frames - len(arr)))
                outdata[:, 0] = arr[:frames].astype(np.float32) / 32768.0
            except asyncio.QueueEmpty:
                outdata[:, 0] = 0.0

        self._stream = sd.OutputStream(
            device=self.device,
            channels=AUDIO_CHANNELS,
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=2048,
            dtype="int16",
            callback=callback,
        )
        with self._stream:
            while self._running:
                await asyncio.sleep(0.05)

    async def play(self, mp3_data: bytes) -> None:
        pcm = decode_mp3(mp3_data)
        if not pcm:
            return
        pcm_int16 = np.frombuffer(pcm, dtype="<h")

        # Push to TTS reference buffer for sidetone cancellation
        if self._tts_ref is not None:
            self._tts_ref.append(pcm_int16)

        with self._playback_lock:
            self._is_playing_tts = True

        # Send in chunks of ~50ms worth of samples
        chunk_size = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH // 20  # 50ms
        for i in range(0, len(pcm), chunk_size):
            await self._queue.put(pcm[i:i + chunk_size])

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await self._task

    def stop_immediately(self) -> None:
        """Stop any TTS playback in progress. Used on barge-in.

        Synchronous so it can be called from the asyncio loop without await.
        Drains the speaker queue and stops the output stream temporarily.
        """
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        with self._playback_lock:
            self._is_playing_tts = False
        # Clear the TTS ref buffer so the mic doesn't subtract stale data
        if self._tts_ref is not None:
            self._tts_ref.clear()


# ── WebSocket client ─────────────────────────────────────────────────

def resolve_device(
    requested: Optional[int],
    kind: str,
    inputs: dict,
    outputs: dict,
) -> Optional[int]:
    """Resolve an audio device index. Returns None to use system default.

    Args:
        requested: device index from env var, or None for auto-detect
        kind: "input" or "output"
        inputs/outputs: dicts from get_devices()
    """
    devs = inputs if kind == "input" else outputs

    # 1. Explicit env var — must be a valid index
    if requested is not None:
        if requested in devs:
            return requested
        logger.error(
            "Requested %s device %d not found. Available: %s",
            kind, requested, sorted(devs.keys()),
        )
        raise SystemExit(1)

    # 2. Try system default
    try:
        default_idx = sd.default.device[0 if kind == "input" else 1]
        if default_idx >= 0 and default_idx in devs:
            logger.info("Using system default %s device: %d (%s)",
                       kind, default_idx, devs[default_idx].get("name", "?"))
            return default_idx
    except Exception:
        pass

    # 3. Fall back to first available
    if devs:
        first_idx = sorted(devs.keys())[0]
        logger.warning("No system default for %s — using first available: %d (%s)",
                      kind, first_idx, devs[first_idx].get("name", "?"))
        return first_idx

    return None


async def run_client() -> None:
    import websockets

    # Auto-select devices
    inputs, outputs = get_devices()

    # Log all available devices for debugging
    logger.info("Available input devices: %s",
                {i: d.get("name", "?") for i, d in inputs.items()})
    logger.info("Available output devices: %s",
                {i: d.get("name", "?") for i, d in outputs.items()})

    # Resolve from env vars (JARVIS_INPUT_DEVICE, JARVIS_OUTPUT_DEVICE) or auto-detect
    env_input = os.environ.get("JARVIS_INPUT_DEVICE")
    env_output = os.environ.get("JARVIS_OUTPUT_DEVICE")
    input_device = resolve_device(
        int(env_input) if env_input else None,
        "input", inputs, outputs,
    )
    output_device = resolve_device(
        int(env_output) if env_output else None,
        "output", inputs, outputs,
    )

    logger.info("JARVIS Voice Client → %s", WS_URL)
    if input_device is not None:
        logger.info("Input device: %d (%s)", input_device, inputs.get(input_device, {}).get("name", "?"))
    else:
        logger.info("Input device: <system default>")
    if output_device is not None:
        logger.info("Output device: %d (%s)", output_device, outputs.get(output_device, {}).get("name", "?"))
    else:
        logger.info("Output device: <system default>")

    # ── Barge-in state (shared between mic capture, speaker, and main loop) ──
    tts_ref = TTSRefBuffer()

    speaker = Speaker(device=output_device)
    speaker.attach_tts_ref(tts_ref)
    await speaker.start()

    mic = MicCapture(device=input_device)
    mic.attach_tts_ref(tts_ref)
    mic.start()

    # Server-driven TTS state: the server tells us when TTS is starting
    # (via `speaking` message) and when the turn is done (via `done`).
    # We use this to gate barge-in detection: only fire barge-in while
    # the server says AI is speaking.
    server_says_ai_speaking = False
    barge_in_fired = False  # latched until server acks

    reconnect_delay = 1.0

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                logger.info("WebSocket connected")
                reconnect_delay = 1.0

                # Send mic frames
                async def send_frames():
                    while True:
                        frame = mic.read()
                        if frame is None:
                            await asyncio.sleep(0.01)
                            continue
                        try:
                            await ws.send(frame)
                        except Exception:
                            break

                # Barge-in watcher: detect user voice on cleaned mic while
                # AI is speaking. Uses the cleaned mic (post-sidetone-cancellation)
                # so the AI's own audio doesn't trigger false positives.
                async def barge_in_watcher():
                    nonlocal barge_in_fired
                    while True:
                        await asyncio.sleep(0.02)  # poll ~50Hz
                        if not server_says_ai_speaking:
                            barge_in_fired = False
                            continue
                        if barge_in_fired:
                            continue
                        # Only fire barge-in on the *cleaned* mic RMS
                        # (after sidetone subtraction, the AI's voice is gone)
                        if mic.is_sidetoning and mic.last_clean_rms > BARGE_IN_RMS:
                            barge_in_fired = True
                            logger.info(
                                "🚨 Barge-in! clean RMS=%.0f > %d — interrupting AI",
                                mic.last_clean_rms, BARGE_IN_RMS,
                            )
                            try:
                                await ws.send(json.dumps({"type": "barge_in"}))
                            except Exception as e:
                                logger.warning("Failed to send barge_in: %s", e)
                            # Stop local TTS immediately
                            speaker.stop_immediately()

                # Receive server messages + audio
                async def recv_messages():
                    nonlocal server_says_ai_speaking, barge_in_fired
                    async for msg in ws:
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            t = data.get("type", "")
                            if t == "vad_state":
                                s = data.get("state", "")
                                logger.info({"idle": "🎙 Listening...", "processing": "⏳ Thinking...", "speaking": "🔊 Speaking..."}.get(s, s))
                            elif t == "transcript":
                                logger.info(f'🗣 "{data.get("text", "")}" (STT {data.get("stt_ms", 0):.0f}ms)')
                            elif t == "response_done":
                                logger.info("💬 Response ready ({:.0f}ms)".format(data.get("llm_ms", 0)))
                            elif t == "speaking":
                                logger.info("🔊 Speaking...")
                                server_says_ai_speaking = True
                            elif t == "barge_in_ack":
                                logger.info("✅ Server acked barge-in")
                                server_says_ai_speaking = False
                                barge_in_fired = False
                                # Belt-and-suspenders: make sure local TTS is stopped
                                speaker.stop_immediately()
                            elif t == "done":
                                total = data.get("total_ms", 0)
                                stt = data.get("stt_ms", 0)
                                llm = data.get("llm_ms", 0)
                                tts = data.get("tts_ms", 0)
                                logger.info(f"✅ Turn complete — STT:{stt}ms LLM:{llm}ms TTS:{tts}ms Total:{total}ms")
                                server_says_ai_speaking = False
                                barge_in_fired = False
                            elif t == "error":
                                logger.error("⚠ %s", data.get("text", ""))
                        elif isinstance(msg, bytes):
                            await speaker.play(msg)

                await asyncio.gather(send_frames(), recv_messages(), barge_in_watcher())

        except Exception as e:
            logger.error("WS error: %s — reconnecting in %.0fs", e, reconnect_delay)
            server_says_ai_speaking = False
            barge_in_fired = False
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)


# ── Entry point ───────────────────────────────────────────────────────

def main():
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logger.info("Shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()