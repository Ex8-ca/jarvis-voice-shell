"""
JARVIS Voice Client — connects to jarvis_ws_gateway.py on .3

- Captures mic from default PipeWire source (Bluetooth headset YYK-Q16)
- Sends PCM frames over WebSocket to .3:6790
- Receives TTS MP3 chunks and plays through default PipeWire sink (Bluetooth headset)

Usage:
    python3 jarvis_voice_client.py [--host 192.168.1.3] [--port 6790]

Dependencies (in venv):
    sounddevice, numpy, websockets, miniaudio
"""

from __future__ import annotations

import asyncio
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

WS_HOST = os.environ.get("JARVIS_WS_HOST", "192.168.1.3")
WS_PORT = int(os.environ.get("JARVIS_WS_PORT", "6790"))
WS_URL = f"ws://{WS_HOST}:{WS_PORT}/ws"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE * 63 // 1000  # 1008
BYTES_PER_FRAME = SAMPLES_PER_FRAME * AUDIO_SAMPLE_WIDTH  # 2016


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
    """Capture mic via sounddevice, put PCM frames in a thread-safe queue."""

    def __init__(self, device: Optional[int] = None, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.device = device
        self.sample_rate = sample_rate
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

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
            # Flatten (frames, 1) → mono bytes
            pcm = bytes(indata.astype("<h").tobytes())
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
    """Play PCM chunks via sounddevice output stream."""

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._stream: Optional[sd.OutputStream] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

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
        # Send in chunks of ~50ms worth of samples
        chunk_size = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH // 20  # 50ms
        for i in range(0, len(pcm), chunk_size):
            await self._queue.put(pcm[i:i + chunk_size])

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await self._task


# ── WebSocket client ─────────────────────────────────────────────────

async def run_client() -> None:
    import websockets

    # Auto-select devices
    inputs, outputs = get_devices()
    # PipeWire default = index 19 on this machine (128 in/out)
    input_device = 19  # PipeWire default (routes to BT mic or built-in)
    output_device = 19  # PipeWire default (routes to BT speaker)

    logger.info("JARVIS Voice Client → %s", WS_URL)
    logger.info("Input device: %d (%s)", input_device, inputs.get(input_device, {}).get("name", "?"))
    logger.info("Output device: %d (%s)", output_device, outputs.get(output_device, {}).get("name", "?"))

    speaker = Speaker(device=output_device)
    await speaker.start()

    mic = MicCapture(device=input_device)
    mic.start()

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

                # Receive server messages + audio
                async def recv_messages():
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
                            elif t == "done":
                                logger.info("✅ Turn complete")
                            elif t == "error":
                                logger.error("⚠ %s", data.get("text", ""))
                        elif isinstance(msg, bytes):
                            await speaker.play(msg)

                await asyncio.gather(send_frames(), recv_messages())

        except Exception as e:
            logger.error("WS error: %s — reconnecting in %.0fs", e, reconnect_delay)
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