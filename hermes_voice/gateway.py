"""
hermes-voice gateway — browser UI + WebSocket server.

Pipeline (one turn): browser mic → WebSocket → EnergyVAD → local Whisper STT →
LLM (Groq/DeepSeek/etc.) → Edge TTS → MP3 → WebSocket → browser speakers.

The mic is hard-muted while the AI is speaking (prevents echo loop).
The AI's response is generated as MP3 chunks and streamed to the client
as they arrive, with the final `done` message triggering playback.
"""
import logging
import os
import sys
import json
from pathlib import Path
import tempfile

import httpx
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

# Plugin-local imports (works when running uvicorn from the plugin dir)
_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from vad import EnergyVAD, VADState, rms_int16  # noqa: E402
from persona import _load_voice_prompt  # noqa: E402
from memory import append_user, append_assistant, append_tool  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hermes-voice.gateway")

# Load .env BEFORE importing llm — llm.py captures env vars at module-load
# time, so the .env has to be in os.environ by then. We use override=False
# so existing process env (from systemd, shell export, etc.) takes priority.
try:
    from dotenv import load_dotenv
    for _env_candidate in [
        Path(__file__).resolve().parent.parent / ".env",  # repo root .env
        Path(__file__).resolve().parent / ".env",         # plugin-local .env
        Path.home() / ".hermes" / "hermes-voice.env",     # user-level .env
    ]:
        if _env_candidate.exists():
            load_dotenv(_env_candidate, override=False)
            logger.info(f"Loaded env from {_env_candidate}")
            break
except ImportError:
    pass

from llm import pick_llm, stream_chat  # noqa: E402
from hermes_voice import tools as _tools_pkg  # noqa: E402  (triggers tool self-registration)
from hermes_voice.tools import REGISTRY as TOOL_REGISTRY  # noqa: E402
from hermes_voice.tools import dispatch as dispatch_tool  # noqa: E402
from hermes_voice.tools import pick_filler  # noqa: E402
from hermes_voice.tools import parse_tool_call, strip_tool_call  # noqa: E402

app = FastAPI(title="Hermes Voice Web")


# Voice system prompt is loaded by the `persona` module (see persona.py).
# Resolution order: HERMES_VOICE_PROMPT_FILE → ~/.hermes/VOICE.md → ~/.hermes/SOUL.md
# → ~/.hermes/USER.md (optional) → generic Hermes persona.


def _build_tool_list_section() -> str:
    """Render the available tools for the LLM's system prompt.

    Only emitted if at least one tool is registered. The LLM is told
    the text-based [[TOOL:...]] syntax, the tool priority order, and
    the available tools with their arguments.
    """
    tools = TOOL_REGISTRY.list()
    if not tools:
        return ""

    lines = [
        "",
        "## Tools",
        "",
        "You have access to the following tools. To call one, output exactly:",
        "",
        "    [[TOOL:tool_name arg1=value1 arg2=value2]]",
        "",
        "Then stop and wait for the result. The system will run the tool and",
        "feed the result back to you as your next turn. Do not write any other",
        "text in the turn that contains a tool call.",
        "",
        "Tool priority (highest first): the system tries the first tool; if it",
        "returns nothing useful, it falls through to the next.",
        "",
    ]
    for t in tools:
        lines.append(f"### {t.name} (priority {t.priority})")
        lines.append(t.description)
        if t.examples:
            lines.append(f"Example: {t.examples[0]}")
        lines.append("")

    return "\n".join(lines)


async def _run_tool_loop(
    response_text: str,
    messages: list,
    max_tok: int,
    *,
    on_token=None,
    filler_tts_to_ws=None,
    max_rounds: int = 3,
    on_tool_result=None,
) -> str:
    """Run the LLM's tool-call loop until we get a final answer.

    The LLM may emit [[TOOL:...]] syntax. For each tool call:
    1. Pick a filler phrase and (optionally) TTS it to the WebSocket
    2. Dispatch the tool (with priority fallback)
    3. Feed the result back to the LLM as a follow-up turn
    4. Stream the LLM's follow-up response via `on_token` (if provided)

    Returns the cleaned, final response text (with all tool calls stripped).
    Loops at most `max_rounds` times to prevent infinite tool-call chains.

    Args:
        response_text: The LLM's first-turn response (may contain a tool call)
        messages: The original messages array (system + user). Follow-up
            turns are appended to this for context.
        max_tok: max_tokens for the follow-up LLM call.
        on_token: Optional async callback(str) called for each streamed token
            of the follow-up response. If None, the response is collected
            silently (used by the chat HTTP endpoint).
        filler_tts_to_ws: Optional WebSocket to send `filler` JSON events
            and filler-phrase MP3 bytes to. If None, no filler TTS is
            generated (chat endpoint).
        on_tool_result: Optional async callback(name: str, text: str) called
            for each tool result, BEFORE the LLM follow-up call. Used by the
            gateway to write tool results to voice memory.
    """
    import time

    for tool_round in range(max_rounds):
        parsed = parse_tool_call(response_text)
        if not parsed:
            return response_text

        tool_name, tool_kwargs = parsed
        logger.info(f"Tool call (round {tool_round + 1}): {tool_name}({tool_kwargs})")

        # Send filler phrase to TTS immediately (masks tool latency)
        filler = pick_filler()
        if filler_tts_to_ws is not None:
            try:
                await filler_tts_to_ws.send_json({"type": "filler", "text": filler})
            except Exception:
                pass
            try:
                import edge_tts
                communicate = edge_tts.Communicate(
                    text=filler, voice="en-GB-RyanNeural", rate="+0%"
                )
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        await filler_tts_to_ws.send_bytes(chunk["data"])
            except Exception as e:
                logger.debug(f"Filler TTS failed (non-fatal): {e}")

        # Run the tool
        tool_start = time.perf_counter()
        tool_result = await dispatch_tool(tool_name, tool_kwargs, fallback=True, timeout_s=15.0)
        tool_ms = (time.perf_counter() - tool_start) * 1000
        logger.info(
            f"Tool '{tool_result.source or tool_name}' done in {tool_ms:.0f}ms: "
            f"{len(tool_result.text)} chars"
        )

        # Notify caller (e.g. gateway writing to voice memory) of tool result
        if on_tool_result is not None and not tool_result.is_empty():
            try:
                await on_tool_result(
                    tool_result.source or tool_name,
                    tool_result.text,
                )
            except Exception:
                logger.exception("on_tool_result callback failed (non-fatal)")

        # If the tool produced nothing useful (and we have no more fallbacks
        # in the chain), just use the LLM's surrounding text and skip the
        # follow-up turn.
        surrounding_text = strip_tool_call(response_text).strip()
        if tool_result.is_empty() and not surrounding_text:
            tool_result_text = (
                f"[Tool {tool_name} returned no results. "
                "Answer from your own knowledge, briefly.]"
            )
        elif tool_result.is_empty():
            response_text = surrounding_text
            return response_text
        else:
            tool_result_text = tool_result.text

        # Feed the tool result back to the LLM as a follow-up turn
        try:
            follow_messages = messages + [
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n{tool_result_text}\n\n"
                        "Now answer the original question briefly."
                    ),
                },
            ]
            response_text = ""
            async for content in stream_chat(follow_messages, max_tokens=max_tok):
                if not content:
                    continue
                response_text += content
                if on_token is not None:
                    await on_token(content)
        except Exception:
            logger.exception("Follow-up LLM call failed")
            return surrounding_text or "I ran into a problem while looking that up."

    return response_text

# Local Whisper STT server
WHISPER_URL = os.getenv("WHISPER_URL", "http://127.0.0.1:9001/v1/audio/transcriptions")

# TTS cache directory
CACHE_DIR = Path.home() / ".cache" / "hermes-voice" / "tts_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the browser UI from static/index.html."""
    static_dir = Path(__file__).resolve().parent / "static"
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>hermes-voice: static/index.html not found</h1>",
            status_code=500,
        )
    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── STT proxy: browser audio → local Whisper ─────────────────────────

@app.post("/stt")
async def stt_transcribe(file: UploadFile = File(...)):
    """Proxy audio to local Faster Whisper server, return transcription."""
    audio_bytes = await file.read()

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                WHISPER_URL,
                files={"file": (file.filename or "audio.webm", audio_bytes, file.content_type or "audio/webm")},
                data={"model": "whisper-1", "response_format": "json"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"text": "", "error": "Whisper server not ready — model still loading. Try again in a minute."}
    except Exception as e:
        logger.exception("STT proxy error")
        return {"text": "", "error": str(e)}


# ── WebSocket voice pipeline with VAD ─────────────────────────────────

@app.websocket("/ws")
async def voice_websocket(ws: WebSocket):
    """Real-time voice pipeline over WebSocket with streaming STT + LLM.
    
    Browser streams raw PCM → EnergyVAD detects speech.
    WHILE user is speaking: interim Whisper STT every ~500ms.
    When speech ends: final STT → streaming LLM tokens → TTS audio.
    New speech cancels any in-progress response.
    
    Protocol (server → browser):
      {type: 'vad_state', state: 'idle'|'primed'|'speaking'|'processing'}
      {type: 'interim_transcript', text: '...'}          — partial STT during speech
      {type: 'transcript', text: '...', stt_ms: N}       — final STT
      {type: 'token', text: '...'}                        — streaming LLM token
      {type: 'response_complete', text: '...', llm_ms: N} — LLM done, full text
      {type: 'speaking'}                                   — TTS starting
      binary audio/mp3 chunks                              — TTS streaming
      {type: 'done'}                                       — turn complete
      {type: 'error', text: '...'}
    """
    import time, edge_tts, hashlib, struct, asyncio, json as _json, wave

    await ws.accept()
    logger.info("WebSocket voice connected (streaming mode)")

    FRAME_MS = 63
    SAMPLES_PER_FRAME = 16000 * FRAME_MS // 1000  # 1008

    # VAD threshold. Default 1500 to ignore ambient room noise / TV / music.
    # Tune via HERMES_VAD_ENERGY_THRESHOLD in .env. Browser wake-word uses
    # 1500 too — matching that gives consistent behavior across web + client.
    import os as _os
    _vad_threshold = int(_os.environ.get("HERMES_VAD_ENERGY_THRESHOLD", "1500"))

    vad = EnergyVAD(
        energy_threshold=_vad_threshold,
        start_frames=3,
        end_silence_frames=5,   # 5 × 63ms = 315ms silence (was 11 = 693ms)
        pre_roll_frames=5,
    )

    processing = False
    current_task: asyncio.Task | None = None

    # Interim STT tracking
    speech_buffer: list[bytes] = []
    interim_seq = 0
    last_interim_send = 0.0
    INTERIM_INTERVAL = 0.5  # seconds between interim Whisper calls

    async def send_interim_stt(audio_data: bytes, seq: int):
        """Send partial audio to Whisper for interim transcription."""
        if len(audio_data) < 16000:  # min 1 second of audio
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            w = wave.open(tmp_path, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio_data)
            w.close()

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                with open(tmp_path, "rb") as f:
                    stt_resp = await client.post(
                        WHISPER_URL,
                        files={"file": ("audio.wav", f, "audio/wav")},
                        data={"model": "whisper-1", "response_format": "json"},
                    )
                stt_resp.raise_for_status()
                stt_data = stt_resp.json()
            text = stt_data.get("text", "").strip()
            if text and seq == interim_seq:  # only if still current
                try:
                    await ws.send_json({"type": "interim_transcript", "text": text})
                except WebSocketDisconnect:
                    pass
        except Exception:
            pass  # interim failures are non-critical
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def receive_loop():
        """Continuously receive PCM chunks and control messages from browser."""
        nonlocal speech_buffer, interim_seq, last_interim_send, processing, current_task

        try:
            while True:
                msg = await ws.receive()

                if msg["type"] == "websocket.disconnect":
                    break

                # Text message = control plane (barge_in, etc.)
                if "text" in msg:
                    try:
                        ctrl = json.loads(msg["text"])
                    except Exception:
                        continue
                    if ctrl.get("type") == "barge_in":
                        # User spoke while AI was responding — cancel in-flight task
                        logger.info(f"Barge-in received: current_task.done()={current_task.done() if current_task else 'no task'}, processing={processing}")
                        if current_task and not current_task.done():
                            current_task.cancel()
                            logger.info("Barge-in: cancelled in-flight LLM/TTS task")
                            try:
                                await ws.send_json({"type": "barge_in_ack"})
                            except Exception:
                                pass
                        # Reset VAD so we start fresh for the new utterance
                        vad.reset()
                        processing = False
                        try:
                            await ws.send_json({"type": "vad_state", "state": "idle"})
                        except Exception:
                            pass
                    continue

                # Binary = PCM audio chunk
                data = msg.get("bytes")
                if not data:
                    logger.info(f"Non-binary msg: {msg.get('type')}")
                    continue
                if len(data) < SAMPLES_PER_FRAME * 2:
                    logger.info(f"Skipping small chunk: {len(data)}B")
                    continue

                old_state = vad.state.value
                segment = vad.process(data)
                new_state = vad.state.value
                if new_state != old_state:
                    logger.info(f"VAD state: {old_state} → {new_state} (chunk {len(data)}B, RMS={rms_int16(data)})")

                if new_state != old_state:
                    await ws.send_json({"type": "vad_state", "state": new_state})

                # During speech: send periodic interim STT
                if vad.state == VADState.SPEAKING:
                    speech_buffer.append(data)
                    now = time.monotonic()
                    if now - last_interim_send > INTERIM_INTERVAL:
                        last_interim_send = now
                        interim_seq += 1
                        audio_snapshot = b"".join(speech_buffer)
                        asyncio.create_task(send_interim_stt(audio_snapshot, interim_seq))

                if segment:
                    # Speech segment complete — cancel any in-progress processing
                    if current_task and not current_task.done():
                        current_task.cancel()
                        logger.info("Cancelled previous processing task — new speech detected")

                    # Invalidate pending interim requests
                    interim_seq += 1
                    # Use accumulated speech buffer if we have it, otherwise VAD segment
                    final_audio = b"".join(speech_buffer) if speech_buffer else segment
                    speech_buffer = []

                    processing = True
                    await ws.send_json({"type": "vad_state", "state": "processing"})
                    current_task = asyncio.create_task(process_segment(final_audio))
                    def _on_done(t):
                        if t.cancelled():
                            logger.warning("process_segment was CANCELLED — TTS may not have run")
                        elif t.exception():
                            logger.exception(f"process_segment raised: {t.exception()}")
                        else:
                            logger.info("process_segment completed normally")
                    current_task.add_done_callback(_on_done)

                if vad.state == VADState.IDLE:
                    speech_buffer = []

        except WebSocketDisconnect:
            pass

    async def process_segment(audio_data: bytes):
        """Run one speech segment: STT → streaming LLM → TTS."""
        nonlocal processing

        t0 = time.perf_counter()
        audio_len_s = len(audio_data) // 2 // 16000
        audio_rms_val = rms_int16(audio_data) if audio_data else 0
        logger.info(f"process_segment: {len(audio_data)} bytes (~{audio_len_s:.1f}s), RMS={audio_rms_val}")

        if audio_rms_val < 50:
            await ws.send_json({"type": "error", "text": "Audio level too low"})
            processing = False
            return

        # ── Step 1: STT via Whisper ────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            w = wave.open(tmp_path, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio_data)
            w.close()

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                with open(tmp_path, "rb") as f:
                    stt_resp = await client.post(
                        WHISPER_URL,
                        files={"file": ("audio.wav", f, "audio/wav")},
                        data={"model": "whisper-1", "response_format": "json"},
                    )
                stt_resp.raise_for_status()
                stt_data = stt_resp.json()
            transcript = stt_data.get("text", "").strip()
        except httpx.ConnectError:
            await ws.send_json({"type": "error", "text": "Whisper not ready — model loading"})
            processing = False
            return
        except Exception:
            logger.exception("STT error")
            processing = False
            return
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if not transcript:
            await ws.send_json({"type": "error", "text": "No speech detected"})
            processing = False
            return

        stt_ms = (time.perf_counter() - t0) * 1000
        await ws.send_json({"type": "transcript", "text": transcript, "stt_ms": stt_ms})

        # Write user turn to voice memory (fire-and-forget — never blocks the
        # voice pipeline. The next LLM call will see this turn as context.)
        try:
            append_user(transcript)
        except Exception:
            logger.exception("voice memory: failed to append user turn")

        # ── Step 2: Streaming LLM (multi-provider) ─────────────────
        llm_start = time.perf_counter()
        base_prompt = _load_voice_prompt()
        tool_section = _build_tool_list_section()
        system_prompt = base_prompt + tool_section
        picked = pick_llm()
        if not picked[0]:
            await ws.send_json({"type": "error", "text": "No LLM configured. Set GROQ_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or LOCAL_LLM_URL in .env"})
            processing = False
            return
        provider_name = picked[3]
        logger.info(f"LLM provider: {provider_name}")
        max_tok = 80  # voice responses are short — cap hard

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}]

        full_response = ""
        try:
            async for content in stream_chat(messages, max_tokens=max_tok):
                if not content:
                    continue
                full_response += content
                await ws.send_json({"type": "token", "text": content})
        except Exception as e:
            logger.exception("LLM streaming error")
            if not full_response:
                await ws.send_json({"type": "error", "text": f"LLM error: {e}"})
                processing = False
                return

        llm_ms = (time.perf_counter() - llm_start) * 1000
        logger.info(f"LLM streaming complete: {len(full_response)} chars in {llm_ms:.0f}ms")

        if not full_response:
            await ws.send_json({"type": "error", "text": "Empty response"})
            processing = False
            return

        await ws.send_json({"type": "response_complete", "text": full_response, "llm_ms": llm_ms})

        # ── Step 2.5: Tool dispatch (if LLM requested a tool) ─────
        # The LLM may have emitted [[TOOL:...]] syntax. If so, the shared
        # _run_tool_loop helper handles dispatch, filler TTS, and follow-up
        # LLM calls. The helper takes an `on_token` callback so we can
        # stream the follow-up response back to the WebSocket.
        async def _ws_on_token(tok: str) -> None:
            try:
                await ws.send_json({"type": "token", "text": tok})
            except Exception:
                pass

        async def _ws_on_tool_result(name: str, text: str) -> None:
            # Write tool result to voice memory (fire-and-forget)
            try:
                append_tool(name, text)
            except Exception:
                logger.exception("voice memory: failed to append tool result")

        full_response = await _run_tool_loop(
            full_response,
            messages,
            max_tok,
            on_token=_ws_on_token,
            filler_tts_to_ws=ws,
            on_tool_result=_ws_on_tool_result,
        )

        # Write assistant turn to voice memory (fire-and-forget). Strips any
        # remaining [[TOOL:...]] markers so the log is human-readable.
        try:
            append_assistant(full_response)
        except Exception:
            logger.exception("voice memory: failed to append assistant turn")

        # ── Step 3: TTS streaming ─────────────────────────────────
        await ws.send_json({"type": "speaking"})

        text_hash = hashlib.sha256(full_response.encode()).hexdigest()[:16]
        cache_path = CACHE_DIR / f"tts_{text_hash}.mp3"

        if cache_path.exists():
            tts_data = cache_path.read_bytes()
            await ws.send_bytes(tts_data)
        else:
            communicate = edge_tts.Communicate(text=full_response, voice="en-GB-RyanNeural", rate="+0%")
            tts_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    tts_chunks.append(chunk["data"])
                    await ws.send_bytes(chunk["data"])
            cache_path.write_bytes(b"".join(tts_chunks))

        await ws.send_json({"type": "done"})
        processing = False
        vad.reset()
        try:
            await ws.send_json({"type": "vad_state", "state": "idle"})
        except WebSocketDisconnect:
            pass

    # Start receive loop
    receive_task = asyncio.create_task(receive_loop())
    try:
        await receive_task
    except asyncio.CancelledError:
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass


# ── Chat endpoints ────────────────────────────────────────────────────

@app.get("/chat")
async def chat_get(text: str):
    return await _process_chat(text)


@app.post("/chat")
async def chat_post(text: str):
    return await _process_chat(text)


async def _process_chat(text: str):
    """Voice-optimized: multi-provider LLM → Edge TTS. Returns text + audio URL."""
    import time
    import edge_tts
    import hashlib

    total_start = time.perf_counter()

    # Step 1: Call LLM (multi-provider via llm.pick_llm / llm.stream_chat)
    bridge_start = time.perf_counter()
    base_prompt = _load_voice_prompt()
    tool_section = _build_tool_list_section()
    system_prompt = base_prompt + tool_section

    picked = pick_llm()
    if not picked[0]:
        return {"error": "No LLM configured. Set GROQ_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or LOCAL_LLM_URL in .env"}
    provider_name = picked[3]
    logger.info(f"Chat endpoint: LLM provider: {provider_name}")
    max_tok = 100  # voice responses are short

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]

    # Collect the full streamed response
    response_text = ""
    async for token in stream_chat(messages, max_tokens=max_tok):
        response_text += token
    bridge_ms = (time.perf_counter() - bridge_start) * 1000

    # Step 1.5: Tool dispatch (chat endpoint) — silently runs tools and feeds
    # the result back to the LLM. The final response_text has the cleaned answer.
    async def _chat_on_tool_result(name: str, tool_text: str) -> None:
        try:
            append_tool(name, tool_text)
        except Exception:
            logger.exception("voice memory: failed to append tool result (chat)")

    response_text = await _run_tool_loop(
        response_text, messages, max_tok, on_tool_result=_chat_on_tool_result
    )

    # Persist this turn to voice memory (chat endpoint also shares the log,
    # so a conversation that starts in chat continues seamlessly if the user
    # switches to voice, or vice versa).
    try:
        append_user(text)
        append_assistant(response_text)
    except Exception:
        logger.exception("voice memory: failed to append chat turn")

    # Step 2: Generate TTS using edge-tts
    tts_start = time.perf_counter()
    text_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"tts_{text_hash}.mp3"

    if not cache_path.exists():
        communicate = edge_tts.Communicate(
            text=response_text,
            voice="en-GB-RyanNeural",
            rate="+0%",
        )
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        audio_data = b"".join(chunks)
        cache_path.write_bytes(audio_data)
    tts_ms = (time.perf_counter() - tts_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "transcript": text,
        "response": response_text,
        "audio_url": f"/audio/{cache_path.name}",
        "bridge_ms": bridge_ms,
        "tts_ms": tts_ms,
        "total_ms": total_ms,
    }


# ── Audio serving ─────────────────────────────────────────────────────

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = CACHE_DIR / filename
    if not filepath.exists():
        return {"error": "Audio not found"}
    return FileResponse(filepath, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)
