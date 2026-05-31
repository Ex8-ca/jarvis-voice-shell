"""OpenAI Realtime voice loop for JARVIS.

This mode deliberately bypasses the fragile local VAD/STT/TTS chain. OpenAI
Realtime owns turn detection, speech input, interruption, and audio output; the
local shell only moves PCM16 between Windows audio devices and the websocket.
"""

from __future__ import annotations

import asyncio
import base64
import json
import inspect
import math
import os
from pathlib import Path
import queue
from dataclasses import dataclass
from typing import Any, Callable


class MissingOpenAIAPIKey(RuntimeError):
    """Raised when OPENAI_API_KEY is required but unavailable."""


@dataclass(frozen=True)
class OpenAIRealtimeVoiceConfig:
    api_key: str
    model: str = "gpt-realtime-mini"
    voice: str = "ash"
    instructions: str = "You are only JARVIS' realtime ears and mouth. Do not answer independently. Wait for Hermes text, then speak it exactly."
    input_device_index: int | None = None
    output_device_index: int | None = None
    device_sample_rate: int = 48000
    api_sample_rate: int = 24000
    channels: int = 1
    output_channels: int = 2
    block_size: int = 2048

    @classmethod
    def from_env(cls, env_files: list[Path] | None = None, **overrides: Any) -> "OpenAIRealtimeVoiceConfig":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        for env_file in (_default_env_files() if env_files is None else env_files):
            if api_key:
                break
            api_key = _read_env_key(env_file, "OPENAI_API_KEY")
        if not api_key:
            raise MissingOpenAIAPIKey(
                "OPENAI_API_KEY is not set. Add it to your environment, then relaunch JARVIS."
            )
        return cls(api_key=api_key, **overrides)


def _default_env_files() -> list[Path]:
    localappdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    return [
        Path.cwd() / ".env",
        localappdata / "hermes" / ".env",
        Path.home() / ".hermes" / ".env",
    ]


def _read_env_key(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def _websocket_headers_kw(connect: Any, headers: dict[str, str]) -> dict[str, dict[str, str]]:
    """Return the header keyword accepted by the installed websockets version."""
    params = inspect.signature(connect).parameters
    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}


def _websocket_connect_kwargs(connect: Any, headers: dict[str, str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = _websocket_headers_kw(connect, headers)
    kwargs["open_timeout"] = 10
    kwargs["max_size"] = None
    return kwargs


def _realtime_headers(api_key: str) -> dict[str, str]:
    """Headers for GA Realtime websocket connections."""
    return {"Authorization": f"Bearer {api_key}"}


def _session_update_payload(config: OpenAIRealtimeVoiceConfig) -> dict[str, Any]:
    """GA Realtime session.update payload."""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": config.model,
            "instructions": config.instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": config.api_sample_rate},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": False,
                        "interrupt_response": True,
                    },
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": config.api_sample_rate},
                    "voice": config.voice,
                },
            },
        },
    }


def _should_cancel_response(*, response_active: bool) -> bool:
    """Only cancel when OpenAI has an active assistant response.

    Server VAD fires speech_started for user speech even when the assistant is
    idle. Sending response.cancel in that state produces
    `Cancellation failed: no active response found`.
    """
    return response_active


def _mono_to_output_channels(data: bytes, channels: int) -> bytes:
    if channels == 1 or not data:
        return data
    import numpy as np

    mono = np.frombuffer(data, dtype="<i2")
    expanded = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    return expanded.astype("<i2").tobytes()


def _pull_ordered_audio(
    audio_out: queue.Queue[bytes | None], pending: bytearray, needed: int
) -> tuple[bytes, bytearray]:
    """Pull exactly needed bytes without reordering leftovers.

    Re-queueing a leftover partial chunk puts it behind newer chunks and corrupts
    audio order. Keep leftovers in a private bytearray instead.
    """
    while len(pending) < needed:
        try:
            chunk = audio_out.get_nowait()
        except queue.Empty:
            break
        if chunk is None:
            break
        pending.extend(chunk)
    data = bytes(pending[:needed])
    del pending[:needed]
    if len(data) < needed:
        data += b"\x00" * (needed - len(data))
    return data, pending


def _speak_text_payload(text: str) -> dict[str, Any]:
    return {
        "type": "response.create",
        "response": {
            "instructions": (
                "You are only the voice output layer for Hermes Agent. "
                "Do not answer independently, add facts, or improvise. "
                "Speak exactly this Hermes response:\n" + text
            ),
        },
    }


class OpenAIRealtimeVoiceClient:
    """Tiny websocket client for OpenAI Realtime audio I/O."""

    def __init__(
        self,
        config: OpenAIRealtimeVoiceConfig,
        on_status: Callable[[str], None] | None = None,
        brain_bridge: Any | None = None,
    ):
        self.config = config
        self._on_status = on_status or (lambda message: None)
        self._brain_bridge = brain_bridge
        self._input_rate_state = None
        self._output_rate_state = None

    def _rate_convert(self, data: bytes, src_rate: int, dst_rate: int, *, output: bool = False) -> bytes:
        if src_rate == dst_rate or not data:
            return data
        import numpy as np

        samples = np.frombuffer(data, dtype="<i2")
        if len(samples) == 0:
            return b""
        expected_len = max(1, round(len(samples) * dst_rate / src_rate))
        try:
            from scipy.signal import resample_poly

            divisor = math.gcd(src_rate, dst_rate)
            up = dst_rate // divisor
            down = src_rate // divisor
            converted = resample_poly(samples.astype(np.float32), up, down)
        except ImportError:
            # Keep OpenAI voice mode usable on lean installs without scipy.
            # Linear interpolation is sufficient for speech I/O and preserves
            # pitch far better than the removed audioop.ratecv dependency path.
            old_positions = np.arange(len(samples), dtype=np.float32)
            new_positions = np.linspace(0, len(samples) - 1, expected_len, dtype=np.float32)
            converted = np.interp(new_positions, old_positions, samples.astype(np.float32))
        if len(converted) > expected_len:
            converted = converted[:expected_len]
        elif len(converted) < expected_len:
            converted = np.pad(converted, (0, expected_len - len(converted)))
        return np.clip(converted, -32768, 32767).astype("<i2").tobytes()

    async def _ask_brain(self, transcript: str) -> str:
        if self._brain_bridge is None:
            return transcript
        self._on_status(f"Hermes heard: {transcript}")
        response = await self._brain_bridge.send(transcript)
        return response.strip()

    async def run(self) -> None:
        try:
            import sounddevice as sd
            import websockets
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("OpenAI voice mode requires sounddevice and websockets") from exc

        url = f"wss://api.openai.com/v1/realtime?model={self.config.model}"
        headers = _realtime_headers(self.config.api_key)
        audio_in: queue.Queue[bytes] = queue.Queue()
        audio_out: queue.Queue[bytes | None] = queue.Queue()

        pending_output = bytearray()

        def input_callback(indata, frames, time_info, status):  # noqa: ANN001
            audio_in.put(bytes(indata))

        def output_callback(outdata, frames, time_info, status):  # noqa: ANN001
            nonlocal pending_output
            data, pending_output = _pull_ordered_audio(audio_out, pending_output, len(outdata))
            outdata[:] = data

        self._on_status("Opening websocket to OpenAI...")
        async with websockets.connect(url, **_websocket_connect_kwargs(websockets.connect, headers)) as ws:
            self._on_status("Connected to OpenAI websocket.")
            await ws.send(json.dumps(_session_update_payload(self.config)))
            self._on_status("Session configured; opening audio streams...")

            async def sender() -> None:
                import websockets as _ws
                try:
                    while True:
                        data = await asyncio.to_thread(audio_in.get)
                        api_data = self._rate_convert(
                            data, self.config.device_sample_rate, self.config.api_sample_rate
                        )
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(api_data).decode("ascii"),
                        }))
                except _ws.exceptions.ConnectionClosedError:
                    pass

            response_active = False

            async def receiver() -> None:
                nonlocal response_active
                async for raw in ws:
                    event = json.loads(raw)
                    event_type = event.get("type")
                    if event_type == "response.created":
                        response_active = True
                    elif event_type in {"response.done", "response.cancelled"}:
                        response_active = False
                    elif event_type == "input_audio_buffer.speech_started":
                        while not audio_out.empty():
                            try:
                                audio_out.get_nowait()
                            except queue.Empty:
                                break
                        if _should_cancel_response(response_active=response_active):
                            await ws.send(json.dumps({"type": "response.cancel"}))
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "").strip()
                        if transcript:
                            hermes_response = await self._ask_brain(transcript)
                            if hermes_response:
                                await ws.send(json.dumps(_speak_text_payload(hermes_response)))
                    elif event_type == "response.output_audio.delta":
                        api_data = base64.b64decode(event.get("delta", ""))
                        device_data = self._rate_convert(
                            api_data,
                            self.config.api_sample_rate,
                            self.config.device_sample_rate,
                            output=True,
                        )
                        audio_out.put(_mono_to_output_channels(device_data, self.config.output_channels))
                    elif event_type == "error":
                        message = event.get("error", {}).get("message", event)
                        raise RuntimeError(f"OpenAI Realtime error: {message}")

            with sd.RawInputStream(
                samplerate=self.config.device_sample_rate,
                channels=self.config.channels,
                dtype="int16",
                device=self.config.input_device_index,
                blocksize=self.config.block_size,
                callback=input_callback,
            ), sd.RawOutputStream(
                samplerate=self.config.device_sample_rate,
                channels=self.config.output_channels,
                dtype="int16",
                device=self.config.output_device_index,
                blocksize=self.config.block_size,
                callback=output_callback,
            ):
                self._on_status("Audio streams open. Speak naturally; Ctrl+C exits.")
                await asyncio.gather(sender(), receiver())
