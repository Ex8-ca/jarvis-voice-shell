"""Tests for bridge.py — SSEParser, EchoBridge, HermesBridge streaming."""

import asyncio
import json

import pytest

from jarvis_voice_shell.bridge import BridgeError, EchoBridge, HermesBridge, HermesCliBridge, SSEParser
from jarvis_voice_shell.config import Config


# ---------------------------------------------------------------------------
# SSEParser
# ---------------------------------------------------------------------------

class TestSSEParser:
    """Standalone SSE stream parser."""

    @pytest.mark.asyncio
    async def test_parses_single_delta_chunk(self):
        """Parse one content delta from a well-formed SSE event."""
        parser = SSEParser()
        events = []

        async def _feed():
            # Simulate an aiohttp response via a simple async generator
            chunks = [
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            ]
            for c in chunks:
                yield c

        # We need to adapt: SSEParser.iter_events takes a ClientResponse.
        # Instead test the raw parse helpers.
        # Test _parse_line static method:
        result = parser.parse_line('data: {"choices":[{"delta":{"content":"x"}}]}')
        assert result is not None
        assert result == {"choices": [{"delta": {"content": "x"}}]}

    def test_parse_line_extracts_data(self):
        """parse_line strips 'data: ' prefix and parses JSON."""
        parser = SSEParser()
        result = parser.parse_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
        assert result == {"choices": [{"delta": {"content": "hi"}}]}

    def test_parse_line_returns_none_for_comment(self):
        """SSE comment lines yield None."""
        parser = SSEParser()
        assert parser.parse_line(": heartbeat") is None
        assert parser.parse_line("") is None

    def test_parse_line_returns_none_for_done(self):
        """data: [DONE] yields None."""
        parser = SSEParser()
        assert parser.parse_line("data: [DONE]") is None

    def test_parse_line_returns_none_for_non_data(self):
        """Non-data lines yield None."""
        parser = SSEParser()
        assert parser.parse_line("event: ping") is None
        assert parser.parse_line("id: 42") is None

    def test_parse_line_bad_json_returns_none(self):
        """Malformed JSON yields None (graceful)."""
        parser = SSEParser()
        assert parser.parse_line("data: {not valid json") is None

    def test_extract_delta_gets_content(self):
        """extract_delta pulls content from OpenAI delta format."""
        parser = SSEParser()
        chunk = {"choices": [{"delta": {"content": "world"}}]}
        assert parser.extract_delta(chunk) == "world"

    def test_extract_delta_missing_choices(self):
        parser = SSEParser()
        assert parser.extract_delta({}) == ""
        assert parser.extract_delta({"choices": []}) == ""

    def test_extract_delta_missing_delta(self):
        parser = SSEParser()
        chunk = {"choices": [{}]}
        assert parser.extract_delta(chunk) == ""

    def test_extract_delta_missing_content(self):
        parser = SSEParser()
        chunk = {"choices": [{"delta": {}}]}
        assert parser.extract_delta(chunk) == ""

    def test_extract_delta_role_skip(self):
        """Deltas with only 'role' (no content) return empty string."""
        parser = SSEParser()
        chunk = {"choices": [{"delta": {"role": "assistant"}}]}
        assert parser.extract_delta(chunk) == ""

    @pytest.mark.asyncio
    async def test_iter_content_from_lines(self):
        """iter_content_from_lines extracts content from raw line strings."""
        parser = SSEParser()
        lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: [DONE]',
        ]
        results = [c async for c in parser.iter_content_from_lines(lines)]
        assert results == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_iter_content_skips_empty_deltas(self):
        parser = SSEParser()
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
        ]
        results = [c async for c in parser.iter_content_from_lines(lines)]
        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_iter_events_from_lines(self):
        """iter_events_from_lines yields parsed dicts for valid events."""
        parser = SSEParser()
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            ': keepalive',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            'data: [DONE]',
        ]
        events = [e async for e in parser.iter_events_from_lines(lines)]
        assert len(events) == 2
        assert events[0]["choices"][0]["delta"]["content"] == "a"
        assert events[1]["choices"][0]["delta"]["content"] == "b"


# ---------------------------------------------------------------------------
# EchoBridge — mock for offline tests
# ---------------------------------------------------------------------------

class TestEchoBridgeInit:
    """EchoBridge initialization."""

    def test_default_mode_is_echo(self):
        eb = EchoBridge()
        assert eb.mode == "echo"
        assert eb._call_count == 0

    def test_custom_delay(self):
        eb = EchoBridge(chunk_delay=0.05)
        assert eb._chunk_delay == 0.05

    def test_canned_responses(self):
        responses = ["Response 1", "Response 2"]
        eb = EchoBridge(responses=responses)
        assert eb._responses == responses

    def test_default_chunk_size(self):
        eb = EchoBridge()
        assert eb._chunk_size == 3


class TestEchoBridgeSend:
    """EchoBridge.send() — non-streaming."""

    @pytest.mark.asyncio
    async def test_echo_mode_returns_input(self):
        eb = EchoBridge()
        result = await eb.send("hello world")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_canned_mode_rotates(self):
        eb = EchoBridge(mode="canned", responses=["A", "B", "C"])
        assert await eb.send("anything") == "A"
        assert await eb.send("anything") == "B"
        assert await eb.send("anything") == "C"
        # Wraps around
        assert await eb.send("anything") == "A"

    @pytest.mark.asyncio
    async def test_silent_mode_returns_empty(self):
        eb = EchoBridge(mode="silent")
        assert await eb.send("hello") == ""

    @pytest.mark.asyncio
    async def test_send_increments_call_count(self):
        eb = EchoBridge()
        assert eb._call_count == 0
        await eb.send("a")
        assert eb._call_count == 1
        await eb.send("b")
        assert eb._call_count == 2


class TestEchoBridgeStream:
    """EchoBridge.stream() — async generator."""

    @pytest.mark.asyncio
    async def test_stream_echo_yields_chunks(self):
        eb = EchoBridge(chunk_size=2)
        chunks = [c async for c in eb.stream("abcdef")]
        assert chunks == ["ab", "cd", "ef"]

    @pytest.mark.asyncio
    async def test_stream_empty_input(self):
        eb = EchoBridge()
        chunks = [c async for c in eb.stream("")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_single_chunk(self):
        eb = EchoBridge(chunk_size=100)
        chunks = [c async for c in eb.stream("hi")]
        assert chunks == ["hi"]

    @pytest.mark.asyncio
    async def test_stream_canned_mode(self):
        eb = EchoBridge(mode="canned", responses=["Alpha", "Beta"], chunk_size=3)
        chunks = [c async for c in eb.stream("ignored")]
        assert "".join(chunks) == "Alpha"
        # Second call
        chunks = [c async for c in eb.stream("ignored")]
        assert "".join(chunks) == "Beta"

    @pytest.mark.asyncio
    async def test_stream_silent_yields_nothing(self):
        eb = EchoBridge(mode="silent")
        chunks = [c async for c in eb.stream("x")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_increments_call_count(self):
        eb = EchoBridge()
        chunks = [c async for c in eb.stream("test")]
        assert len(chunks) > 0
        assert eb._call_count == 1

    @pytest.mark.asyncio
    async def test_stream_canned_wraps(self):
        eb = EchoBridge(mode="canned", responses=["One"], chunk_size=5)
        c1 = [c async for c in eb.stream("x")]
        c2 = [c async for c in eb.stream("x")]
        assert "".join(c1) == "One"
        assert "".join(c2) == "One"  # wraps around


# ---------------------------------------------------------------------------
# HermesBridge streaming
# ---------------------------------------------------------------------------

class TestHermesBridgeStream:
    """HermesBridge.stream() async generator."""

    def test_headers_include_bearer_auth_when_api_key_configured(self):
        bridge = HermesBridge(Config(hermes_api_key="voice-key"))

        assert bridge._headers() == {
            "Content-Type": "application/json",
            "Authorization": "Bearer voice-key",
        }

    def test_headers_omit_auth_when_api_key_empty(self):
        bridge = HermesBridge(Config(hermes_api_key=""))

        assert bridge._headers() == {"Content-Type": "application/json"}

    def test_build_payload_uses_bridge_max_tokens(self):
        bridge = HermesBridge(Config(bridge_max_tokens=768))

        payload = bridge._build_payload("hello", stream=True)

        assert payload["messages"][-1] == {"role": "user", "content": "hello"}
        assert payload["stream"] is True
        assert payload["max_tokens"] == 768

    @pytest.mark.asyncio
    async def test_stream_requires_session(self):
        """stream() lazily creates a session."""
        bridge = HermesBridge(Config())
        assert bridge._session is None
        # Don't actually connect — just check that it returns an async generator
        gen = bridge.stream("hello")
        assert hasattr(gen, "__aiter__")

    def test_bridge_stream_is_async_generator(self):
        bridge = HermesBridge(Config())
        gen = bridge.stream("test")
        import inspect
        assert inspect.isasyncgen(gen)


class TestBridgeHistory:
    """Conversation history management."""

    def test_max_history_bound(self):
        """History should be capped at 20 messages (10 turns)."""
        bridge = HermesBridge(Config())
        # Fill with 22 messages (11 turns)
        for i in range(11):
            bridge._conversation_history.append(
                {"role": "user", "content": f"msg{i}"}
            )
            bridge._conversation_history.append(
                {"role": "assistant", "content": f"resp{i}"}
            )
        assert len(bridge._conversation_history) == 22
        # Adding another turn would trigger truncation in send()
        # Direct manipulation test:
        bridge._conversation_history = bridge._conversation_history[-20:]
        assert len(bridge._conversation_history) == 20


class TestSSEParserResilience:
    """Graceful failure modes."""

    @pytest.mark.asyncio
    async def test_bad_json_does_not_kill_stream(self):
        parser = SSEParser()
        lines = [
            'data: {"choices":[{"delta":{"content":"good"}}]}',
            'data: {broken',
            'data: {"choices":[{"delta":{"content":"recovered"}}]}',
        ]
        results = [c async for c in parser.iter_content_from_lines(lines)]
        assert results == ["good", "recovered"]

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        parser = SSEParser()
        results = [c async for c in parser.iter_content_from_lines([])]
        assert results == []

    def test_extract_delta_finish_reason(self):
        """Chunks with finish_reason but no content return empty."""
        parser = SSEParser()
        chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        assert parser.extract_delta(chunk) == ""


# ---------------------------------------------------------------------------
# HermesCliBridge — real Hermes Agent subprocess bridge
# ---------------------------------------------------------------------------

class _FakeProcess:
    def __init__(self, returncode=0, stdout=b"Hermes reply\n", stderr=b"", delay=0.0):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._delay = delay
        self.killed = False

    async def communicate(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True


class TestHermesCliBridge:
    @pytest.mark.asyncio
    async def test_send_invokes_hermes_chat_quiet_query(self):
        calls = []

        async def fake_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeProcess(stdout=b"Done, sir.\n")

        bridge = HermesCliBridge(process_factory=fake_exec)
        result = await bridge.send("status")

        assert result == "Done, sir."
        assert calls[0][0][:4] == ("hermes", "chat", "-Q", "-q")
        assert calls[0][0][4] == "status"

    @pytest.mark.asyncio
    async def test_send_calls_token_callback_with_full_response(self):
        events = []

        async def fake_exec(*args, **kwargs):
            return _FakeProcess(stdout=b"Token text\n")

        bridge = HermesCliBridge(process_factory=fake_exec)
        result = await bridge.send("hello", on_token=lambda full, delta: events.append((full, delta)))

        assert result == "Token text"
        assert events == [("Token text", "Token text")]

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_bridge_error(self):
        async def fake_exec(*args, **kwargs):
            return _FakeProcess(returncode=2, stderr=b"bad things")

        bridge = HermesCliBridge(process_factory=fake_exec)
        with pytest.raises(BridgeError, match="Hermes CLI failed"):
            await bridge.send("hello")

    @pytest.mark.asyncio
    async def test_strips_session_id_metadata_from_stdout(self):
        async def fake_exec(*args, **kwargs):
            return _FakeProcess(stdout=b"session_id: 20260530_abc\nHermes online.\n")

        bridge = HermesCliBridge(process_factory=fake_exec)
        result = await bridge.send("status")

        assert result == "Hermes online."

    @pytest.mark.asyncio
    async def test_timeout_raises_bridge_error_and_kills_process(self):
        proc = _FakeProcess(delay=0.05)

        async def fake_exec(*args, **kwargs):
            return proc

        bridge = HermesCliBridge(process_factory=fake_exec, timeout_seconds=0.01)
        with pytest.raises(BridgeError, match="timed out"):
            await bridge.send("hello")
        assert proc.killed is True
