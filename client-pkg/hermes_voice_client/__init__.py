"""
hermes-voice-client — Standalone Python client for the hermes_voice gateway.

Self-contained voice client. Captures mic on a client machine, streams raw PCM
over WebSocket to a hermes_voice gateway (running elsewhere on the LAN), and
plays back TTS audio from the gateway.

For single-machine deployments, use the hermes-voice web UI instead.

Usage:
    pip install hermes-voice-client
    hermes-voice-client

Environment:
    HERMES_WS_HOST              Gateway host (default: 192.168.1.3)
    HERMES_WS_PORT              Gateway port (default: 8989)
    HERMES_INPUT_DEVICE         Mic device index (default: auto-detect)
    HERMES_OUTPUT_DEVICE        Speaker device index (default: auto-detect)
    HERMES_LOG_LEVEL            DEBUG / INFO / WARNING (default: INFO)

List audio devices:
    python3 -c "import sounddevice; print(sounddevice.query_devices())"
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

import numpy as np
import sounddevice as sd

__version__ = "0.1.3"

logger = logging.getLogger("hermes-voice-client")


# ── Config ────────────────────────────────────────────────────────────

WS_HOST = os.environ.get("HERMES_WS_HOST", "192.168.1.3")
WS_PORT = int(os.environ.get("HERMES_WS_PORT", "8989"))
WS_URL = f"ws://{WS_HOST}:{WS_PORT}/ws"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE * 63 // 1000  # 1008 samples = 63ms
BYTES_PER_FRAME = SAMPLES_PER_FRAME * AUDIO_SAMPLE_WIDTH  # 2016

# Many output devices (e.g. laptop analog jacks, some HDMI sinks) don't
# natively support 16kHz, which is what the gateway uses. PortAudio's ALSA
# backend will then fail with "paInvalidSampleRate" and the output stream
# silently never starts. Using a universally-supported rate for the speaker
# lets PortAudio transparently resample from 16kHz MP3 → 48kHz PCM.
SPEAKER_SAMPLE_RATE = 48000
SPEAKER_BLOCK_SIZE = 2048


# ── Device enumeration ───────────────────────────────────────────────

def get_devices() -> tuple[dict, dict]:
    """Return (inputs, outputs) dicts of {index: device_info}."""
    devs = sd.query_devices()
    inputs, outputs = {}, {}
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            inputs[i] = d
        if d.get("max_output_channels", 0) > 0:
            outputs[i] = d
    return inputs, outputs


def resolve_device(requested, kind: str, inputs: dict, outputs: dict) -> Optional[int]:
    """Pick an audio device. Explicit env var > system default > first available."""
    devs = inputs if kind == "input" else outputs
    if requested is not None:
        if requested in devs:
            return requested
        logger.error(
            "Requested %s device %d not found. Available: %s",
            kind, requested, sorted(devs.keys()),
        )
        raise SystemExit(1)
    try:
        default_idx = sd.default.device[0 if kind == "input" else 1]
        if default_idx >= 0 and default_idx in devs:
            logger.info(
                "Using system default %s device: %d (%s)",
                kind, default_idx, devs[default_idx].get("name", "?"),
            )
            return default_idx
    except Exception:
        pass
    if devs:
        first_idx = sorted(devs.keys())[0]
        logger.warning(
            "No system default for %s — using first available: %d (%s)",
            kind, first_idx, devs[first_idx].get("name", "?"),
        )
        return first_idx
    return None


# ── Mic capture (background thread → queue) ──────────────────────────

class MicCapture:
    """Capture mic via sounddevice, push PCM frames into a thread-safe queue."""

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
        try:
            dev = sd.query_devices(self.device)
            logger.info(
                "Mic capture: device=%s, rate=%.0f, frames=%d bytes",
                dev.get("name", "default"), self.sample_rate, BYTES_PER_FRAME,
            )
        except Exception:
            logger.info(
                "Mic capture: device=%s, rate=%.0f",
                self.device if self.device is not None else "default",
                self.sample_rate,
            )

    def _on_frame(self, indata, frames, status, _):
        if status:
            logger.debug("capture status: %s", status)
        try:
            self._queue.put_nowait(bytes(indata.astype("<h").tobytes()))
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
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._stream = None


# ── MP3 decode + speaker ─────────────────────────────────────────────

def decode_mp3(mp3_data: bytes) -> bytes:
    """Decode MP3 bytes → 16-bit mono PCM at SPEAKER_SAMPLE_RATE via miniaudio.

    The output is resampled to the speaker's native rate, since most output
    devices (laptop analog jacks, HDMI sinks, Bluetooth) don't accept 16kHz.
    """
    import miniaudio as ma
    try:
        decoded = ma.decode(
            mp3_data,
            output_format=ma.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=SPEAKER_SAMPLE_RATE,
        )
        samples = decoded.samples
        if samples is None or len(samples) == 0:
            return b""
        return bytes(samples.tobytes()) if hasattr(samples, "tobytes") else bytes(samples)
    except Exception as e:
        logger.warning("MP3 decode failed: %s", e)
        return b""


class Speaker:
    """Async queue → sounddevice output stream."""

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._stream: Optional[sd.OutputStream] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._play_loop())
        logger.info(
            "Speaker: device=%s rate=%d",
            self.device if self.device is not None else "default",
            SPEAKER_SAMPLE_RATE,
        )

    async def _play_loop(self) -> None:
        def callback(outdata, frames, status, _):
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
            samplerate=SPEAKER_SAMPLE_RATE,
            blocksize=SPEAKER_BLOCK_SIZE,
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
        # Send in chunks of ~50ms worth of samples (at the speaker's rate).
        chunk_size = SPEAKER_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH // 20
        for i in range(0, len(pcm), chunk_size):
            await self._queue.put(pcm[i:i + chunk_size])

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await self._task


# ── Server message routing ───────────────────────────────────────────

def _handle_text(msg: str) -> None:
    """Pretty-print non-audio messages from the gateway."""
    try:
        data = json.loads(msg)
    except json.JSONDecodeError:
        logger.debug("Non-JSON msg: %r", msg[:80])
        return
    mtype = data.get("type", "?")
    if mtype == "vad_state":
        logger.debug("VAD: %s", data.get("state"))
    elif mtype == "transcript":
        logger.info("Transcript: %s", data.get("text"))
    elif mtype == "token":
        sys.stdout.write(data.get("text", ""))
        sys.stdout.flush()
    elif mtype == "response_complete":
        print()
        logger.info("Response done in %sms", data.get("llm_ms"))
    elif mtype == "speaking":
        logger.info("TTS playing...")
    elif mtype == "error":
        logger.error("Server error: %s", data.get("message"))
    else:
        logger.debug("Server msg: %s", mtype)


# ── Main loop ────────────────────────────────────────────────────────

async def run() -> None:
    """Main entry: connect WebSocket, stream mic, play TTS."""
    import websockets

    inputs, outputs = get_devices()
    logger.info(
        "Inputs: %s",
        {i: d.get("name", "?") for i, d in inputs.items()},
    )
    logger.info(
        "Outputs: %s",
        {i: d.get("name", "?") for i, d in outputs.items()},
    )

    env_input = os.environ.get("HERMES_INPUT_DEVICE")
    env_output = os.environ.get("HERMES_OUTPUT_DEVICE")
    input_device = resolve_device(
        int(env_input) if env_input else None, "input", inputs, outputs
    )
    output_device = resolve_device(
        int(env_output) if env_output else None, "output", inputs, outputs
    )

    logger.info("Hermes Voice Client v%s → %s", __version__, WS_URL)
    logger.info(
        "Input: %s | Output: %s",
        input_device if input_device is not None else "<default>",
        output_device if output_device is not None else "<default>",
    )

    speaker = Speaker(device=output_device)
    await speaker.start()
    mic = MicCapture(device=input_device)
    mic.start()

    reconnect_delay = 1.0
    while True:
        try:
            async with websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                logger.info("WebSocket connected")
                reconnect_delay = 1.0

                async def sender():
                    while True:
                        frame = mic.read()
                        if frame:
                            await ws.send(frame)
                        else:
                            await asyncio.sleep(0.01)

                # Buffer for reassembling fragmented MP3 (TTS streams as many
                # tiny binary frames; miniaudio needs the complete MP3 to decode).
                mp3_buffer = bytearray()
                mp3_playing = False

                async def _flush_mp3():
                    """Decode and play whatever's in the MP3 buffer, then clear it."""
                    nonlocal mp3_buffer
                    if not mp3_buffer:
                        return
                    logger.info(
                        "TTS received: %d bytes MP3, decoding...", len(mp3_buffer)
                    )
                    await speaker.play(bytes(mp3_buffer))
                    mp3_buffer = bytearray()

                async def receiver():
                    nonlocal mp3_buffer, mp3_playing
                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, (bytes, bytearray)):
                            # Accumulate TTS binary frames. Decoding waits for `done`.
                            mp3_buffer.extend(msg)
                        else:
                            # Text frames: routing, status, and turn-end signals.
                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError:
                                logger.debug("Non-JSON msg: %r", msg[:80])
                                continue
                            mtype = data.get("type")
                            if mtype == "vad_state":
                                logger.debug("VAD: %s", data.get("state"))
                            elif mtype == "transcript":
                                logger.info("Transcript: %s", data.get("text"))
                            elif mtype == "token":
                                sys.stdout.write(data.get("text", ""))
                                sys.stdout.flush()
                            elif mtype == "response_complete":
                                print()
                                logger.info(
                                    "Response done in %sms", data.get("llm_ms")
                                )
                            elif mtype == "speaking":
                                logger.info("TTS playing...")
                            elif mtype == "error":
                                # Tolerate both "message" (canonical) and "text"
                                # (legacy) keys — gateway has used both at
                                # different points in its history.
                                err = data.get("message") or data.get("text") or "(no message)"
                                logger.error("Server error: %s", err)
                                # Drop any partial MP3 — TTS aborted.
                                if mp3_buffer:
                                    logger.warning(
                                        "Discarding %d bytes of partial MP3",
                                        len(mp3_buffer),
                                    )
                                    mp3_buffer = bytearray()
                            elif mtype == "done":
                                # Turn complete — play the accumulated TTS.
                                await _flush_mp3()
                            else:
                                logger.debug("Server msg: %s", mtype)

                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())
                done, pending = await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_EXCEPTION
                )
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
        except (OSError, websockets.exceptions.WebSocketException) as e:
            logger.warning(
                "Connection lost: %s. Reconnecting in %.1fs...", e, reconnect_delay
            )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 10.0)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Unexpected error")
            await asyncio.sleep(reconnect_delay)


def main() -> None:
    """CLI entry point."""
    level = os.environ.get("HERMES_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(*_):
        logger.info("Shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(run())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
