"""Streaming HTTP bridge to Hermes Agent.

Sends user transcripts to Hermes via OpenAI-compatible chat completions API.
Receives streaming SSE responses token by token.
Designed for minimal time-to-first-byte (TTFB).

Key design properties:
    - Async (aiohttp) — non-blocking while Hermes streams
    - SSEParser — standalone, testable SSE parsing (OpenAI-compatible)
    - EchoBridge — offline/mock path for tests without real Hermes
    - Callback-based — caller receives tokens as they arrive
    - Connection reuse — single aiohttp session per Bridge lifetime
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import AsyncIterator, Callable, Optional

import aiohttp

from .config import Config
from .latency import TurnLatency

logger = logging.getLogger(__name__)

# Type for token callback: receives the accumulated text so far and the new delta.
TokenCallback = Callable[[str, str], None]


class BridgeError(Exception):
    """Raised when the Hermes bridge returns an error."""


# ---------------------------------------------------------------------------
# SSEParser — standalone, testable SSE stream parser
# ---------------------------------------------------------------------------

class SSEParser:
    """Parse OpenAI-compatible Server-Sent Events streaming response.

    Design:
        - Handles ``data: {json}`` lines and ``data: [DONE]`` termination.
        - Silently skips comments (``: ...``), keepalives, and empty lines.
        - Malformed JSON is logged and skipped — never kills the stream.
        - Extract content deltas via ``extract_delta()`` static method.

    Usage with a real aiohttp response::

        parser = SSEParser()
        full = ""
        async for content in parser.iter_content(response):
            full += content
            print(content, end="", flush=True)
        return full

    For offline testing, use ``iter_content_from_lines(lines)`` which takes
    plain strings instead of a streaming HTTP response.
    """

    @staticmethod
    def parse_line(line: str) -> dict | None:
        """Parse a single SSE text line into a dict, or None.

        Returns None for: empty lines, comments (``: ...``),
        ``data: [DONE]``, non-data fields, and unparseable JSON.
        """
        line = line.strip()
        if not line:
            return None
        if line.startswith(":"):
            return None  # SSE comment / keepalive
        if line == "data: [DONE]":
            return None
        if not line.startswith("data: "):
            return None

        data_str = line[len("data: "):]
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("SSEParser: unparseable chunk: %s", data_str[:100])
            return None

    @staticmethod
    def extract_delta(chunk: dict) -> str:
        """Extract the content delta from an OpenAI-style chunk dict.

        Expected shape::

            {"choices": [{"delta": {"content": "hello"}}]}

        Returns the content string, or ``""`` if the path is missing
        (e.g. role-only deltas, finish_reason-only chunks).
        """
        choices = chunk.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {})
        return delta.get("content", "")

    # -- Async generator over real aiohttp response -----------------------

    async def iter_events(self, response: aiohttp.ClientResponse) -> AsyncIterator[dict]:
        """Yield parsed JSON dicts from an aiohttp streaming response.

        Each SSE event line is parsed; non-data / unparseable lines are skipped.
        """
        async for raw_line in response.content:
            line_str = raw_line.decode("utf-8").strip()
            event = self.parse_line(line_str)
            if event is not None:
                yield event

    async def iter_content(self, response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        """Yield content delta strings from a streaming response."""
        async for event in self.iter_events(response):
            content = self.extract_delta(event)
            if content:
                yield content

    # -- Offline helpers (test-friendly) ----------------------------------

    async def iter_events_from_lines(self, lines: list[str]) -> AsyncIterator[dict]:
        """Offline variant: yield parsed events from a list of SSE line strings."""
        for line in lines:
            event = self.parse_line(line)
            if event is not None:
                yield event

    async def iter_content_from_lines(self, lines: list[str]) -> AsyncIterator[str]:
        """Offline variant: yield content deltas from a list of SSE line strings."""
        async for event in self.iter_events_from_lines(lines):
            content = self.extract_delta(event)
            if content:
                yield content


# ---------------------------------------------------------------------------
# EchoBridge — mock for offline / no-network testing
# ---------------------------------------------------------------------------

class EchoBridge:
    """Offline bridge that echoes input or plays canned responses.

    Modes:
        - ``"echo"`` (default): Streams the input text back as chunks.
        - ``"canned"``: Rotates through a fixed list of responses.
        - ``"silent"``: Returns empty strings (simulates Hermes down).

    Designed as a drop-in replacement for ``HermesBridge`` in tests and
    dry-run mode. Supports both ``send()`` (full response) and ``stream()``
    (async generator) APIs.

    No secrets — purely deterministic mock.
    """

    def __init__(
        self,
        mode: str = "echo",
        responses: list[str] | None = None,
        chunk_size: int = 3,
        chunk_delay: float = 0.0,
    ):
        self.mode = mode
        self._responses = responses or []
        self._chunk_size = max(1, chunk_size)
        self._chunk_delay = chunk_delay
        self._call_count = 0
        self._rotation_index = 0

    # -- Non-streaming API -------------------------------------------------

    async def send(
        self,
        transcript: str,
        on_token: Optional[TokenCallback] = None,
        latency: Optional[TurnLatency] = None,
    ) -> str:
        """Return the full response text (non-streaming API)."""
        import time

        if latency is not None:
            latency.bridge_start = time.perf_counter()

        text = self._pick_response(transcript)

        if latency is not None:
            now = time.perf_counter()
            latency.bridge_first_token = now
            latency.bridge_last_token = now

        if on_token:
            on_token(text, text)

        self._call_count += 1
        return text

    # -- Streaming API -----------------------------------------------------

    async def stream(self, transcript: str) -> AsyncIterator[str]:
        """Yield response chunks for simulating streaming behavior."""
        text = self._pick_response(transcript)

        for i in range(0, len(text), self._chunk_size):
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield text[i:i + self._chunk_size]

        self._call_count += 1

    # -- Internal ----------------------------------------------------------

    def _pick_response(self, _transcript: str) -> str:
        """Select the response text based on mode."""
        if self.mode == "silent":
            return ""
        if self.mode == "canned":
            if not self._responses:
                return ""
            idx = self._rotation_index % len(self._responses)
            self._rotation_index += 1
            return self._responses[idx]
        # "echo" mode (default)
        return _transcript

    # -- Lifecycle ---------------------------------------------------------

    async def reset_conversation(self) -> None:
        self._rotation_index = 0

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HermesCliBridge — subprocess bridge to the real Hermes Agent CLI
# ---------------------------------------------------------------------------

class HermesCliBridge:
    """Bridge that sends each transcript to real Hermes Agent via CLI.

    This is slower than the raw HTTP model bridge but preserves Hermes' system
    prompt, tools, skills, memory, and provider routing. It intentionally
    matches the ``send()``/``close()`` shape used by EchoBridge and HermesBridge.
    """

    def __init__(
        self,
        hermes_command: str = "hermes",
        process_factory=None,
        extra_args: list[str] | None = None,
        timeout_seconds: float = 60.0,
    ):
        self._hermes_command = hermes_command
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._extra_args = extra_args or []
        self._timeout_seconds = timeout_seconds

    def _build_command(self, transcript: str) -> list[str]:
        return [
            self._hermes_command,
            "chat",
            "-Q",
            "-q",
            transcript,
            *self._extra_args,
        ]

    async def send(
        self,
        transcript: str,
        on_token: Optional[TokenCallback] = None,
        latency: Optional[TurnLatency] = None,
    ) -> str:
        import time

        if latency is not None:
            latency.bridge_start = time.perf_counter()

        proc = await self._process_factory(
            *self._build_command(transcript),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            except Exception:
                pass
            raise BridgeError(f"Hermes CLI timed out after {self._timeout_seconds:.0f}s") from exc
        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            raise BridgeError(f"Hermes CLI failed ({proc.returncode}): {err_text[:500]}")

        response = self._clean_stdout(stdout.decode("utf-8", errors="replace"))
        if latency is not None:
            now = time.perf_counter()
            latency.bridge_first_token = now
            latency.bridge_last_token = now
        if on_token and response:
            on_token(response, response)
        return response

    @staticmethod
    def _clean_stdout(stdout: str) -> str:
        """Remove Hermes CLI metadata lines that should never be spoken."""
        lines = []
        for line in stdout.splitlines():
            if line.strip().lower().startswith("session_id:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    async def stream(self, transcript: str) -> AsyncIterator[str]:
        response = await self.send(transcript)
        if response:
            yield response

    async def reset_conversation(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HermesBridge — streaming HTTP client for Hermes Agent
# ---------------------------------------------------------------------------

class HermesBridge:
    """Streaming HTTP client for Hermes Agent."""

    def __init__(self, config: Config):
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._conversation_history: list[dict] = []
        self._parser = SSEParser()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.bridge_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _headers(self) -> dict:
        """Build HTTP headers for the Hermes Gateway API Server."""
        headers = {"Content-Type": "application/json"}
        api_key = getattr(self._config, "hermes_api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # -- Non-streaming API (existing, preserved) ---------------------------

    async def send(
        self,
        transcript: str,
        on_token: Optional[TokenCallback] = None,
        latency: Optional[TurnLatency] = None,
    ) -> str:
        """Send a transcript to Hermes and receive the response.

        Args:
            transcript: The user's spoken text.
            on_token: Called with (full_text_so_far, new_delta) for each token.
            latency: If provided, bridge_start / bridge_first_token / bridge_last_token
                     are stamped in-place.

        Returns:
            The complete response text from Hermes.

        Raises:
            BridgeError: On HTTP errors or JSON parse failures.
        """
        import time

        session = await self._ensure_session()

        payload = self._build_payload(transcript, stream=self._config.bridge_stream)

        if latency is not None:
            latency.bridge_start = time.perf_counter()

        try:
            async with session.post(
                self._config.hermes_bridge_url,
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise BridgeError(
                        f"Hermes returned {response.status}: {body[:500]}"
                    )

                if self._config.bridge_stream:
                    full_text = await self._parse_sse_stream(
                        response, on_token, latency,
                    )
                else:
                    data = await response.json()
                    full_text = data["choices"][0]["message"]["content"]
                    if on_token:
                        on_token(full_text, full_text)
                    if latency is not None:
                        now = time.perf_counter()
                        latency.bridge_first_token = now
                        latency.bridge_last_token = now

        except aiohttp.ClientError as e:
            raise BridgeError(f"Connection to Hermes failed: {e}") from e

        self._update_history(transcript, full_text)
        return full_text

    # -- Streaming API (async generator) -----------------------------------

    async def stream(self, transcript: str) -> AsyncIterator[str]:
        """Stream response chunks from Hermes via async generator.

        Yields content delta strings as they arrive. Use in an async for loop::

            async for chunk in bridge.stream("hello"):
                print(chunk, end="", flush=True)

        Raises:
            BridgeError: On HTTP errors, timeouts, or connection failures.
        """
        session = await self._ensure_session()
        payload = self._build_payload(transcript, stream=self._config.bridge_stream)
        full_text = ""

        try:
            async with session.post(
                self._config.hermes_bridge_url,
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise BridgeError(
                        f"Hermes returned {response.status}: {body[:500]}"
                    )

                if not self._config.bridge_stream:
                    # Non-streaming fallback: yield the full response as one chunk
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    if content:
                        yield content
                        full_text = content
                else:
                    async for chunk in self._parser.iter_content(response):
                        full_text += chunk
                        yield chunk

        except aiohttp.ClientError as e:
            raise BridgeError(f"Connection to Hermes failed: {e}") from e

        self._update_history(transcript, full_text)

    # -- Payload / history helpers -----------------------------------------

    def _build_payload(self, transcript: str, stream: bool | None = None) -> dict:
        """Build the OpenAI-compatible chat completions request payload."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are JARVIS, a voice assistant. You speak concisely — "
                    "2-3 sentences per response. You have a dry British wit. "
                    "You address the user as 'sir'. Keep responses brief and "
                    "conversational, suitable for spoken output."
                ),
            },
            *self._conversation_history,
            {"role": "user", "content": transcript},
        ]
        return {
            "messages": messages,
            "stream": self._config.bridge_stream if stream is None else stream,
            "temperature": 0.7,
            "max_tokens": getattr(self._config, "bridge_max_tokens", 512),
        }

    def _update_history(self, transcript: str, full_text: str) -> None:
        """Append the turn to conversation history, keeping it bounded."""
        self._conversation_history.append({"role": "user", "content": transcript})
        self._conversation_history.append({"role": "assistant", "content": full_text})
        max_history = 20  # 10 turns
        if len(self._conversation_history) > max_history:
            self._conversation_history = self._conversation_history[-max_history:]

    # -- SSE streaming (used by send()) ------------------------------------

    async def _parse_sse_stream(
        self,
        response: aiohttp.ClientResponse,
        on_token: Optional[TokenCallback],
        latency: Optional[TurnLatency],
    ) -> str:
        """Parse Server-Sent Events from Hermes streaming response."""
        import time

        full_text = ""
        first_token = True

        async for content in self._parser.iter_content(response):
            if first_token and latency is not None:
                latency.bridge_first_token = time.perf_counter()
                first_token = False

            full_text += content
            if on_token:
                on_token(full_text, content)

        if latency is not None:
            latency.bridge_last_token = time.perf_counter()

        return full_text

    # -- Lifecycle ---------------------------------------------------------

    async def reset_conversation(self) -> None:
        """Clear conversation history for a fresh session."""
        self._conversation_history.clear()
        logger.info("Conversation history reset.")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Bridge session closed.")
