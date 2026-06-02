# JARVIS Voice Shell

Real-time voice pipeline for AI assistants. Streaming STT → streaming LLM → streaming TTS, with filler phrases that mask latency and a wake-word-free always-on mic. Runs in two modes: single-machine (mic + STT + LLM + TTS in one box, web UI in your browser) or split (mic on one machine, STT/LLM/TTS on another over WebSocket).

> JARVIS is used here as an assistant-style project name. This project is not affiliated with Marvel, Disney, OpenAI, Microsoft, or Nous Research.

## What you get

- **Always-on mic** with energy-based VAD — no push-to-talk, no wake word needed for the web UI
- **Streaming Whisper STT** on the server (Faster-Whisper, `large-v3-turbo` by default)
- **Streaming LLM** — Groq (~150ms first token), DeepSeek, OpenAI, local (Ollama/vLLM), or any OpenAI-compatible proxy. First available key wins.
- **Filler phrases** ("One sec...", "Checking...") play the instant you stop talking, so the AI's pause feels like thinking, not lag
- **Edge TTS** streams MP3 chunks back to the client as they synthesize (no waiting for the full audio)
- **Barge-in** — interrupt the AI mid-response by talking over it
- **Two deployment modes** — single-machine (web UI) or split-architecture (Python client on your desktop, gateway on a server)
- **Docker** — one-command deploy with optional GPU

## Architecture

### Single-machine mode

Mic + STT + LLM + TTS all on one computer. The browser opens `http://localhost:8989`, grants mic permission, and you're talking to the AI.

```
Browser mic ──► Web UI (:8989) ──► Whisper (:9001) ──► LLM API ──► Edge TTS
                       │                                                    │
                       └──────────────── TTS audio back ──────────────────┘
                       │
                  Browser speakers
```

### Split-architecture mode

Mic + speakers on your desktop; STT + LLM + TTS on a server. Useful when the desktop is underpowered or you want a "voice satellite" setup. The Python client (`jarvis_voice_client.py`) does the audio I/O locally; everything heavy runs remotely.

```
Desktop (.2)                                              Server (.3)
───────────                                               ─────────
Microphone ──► Python client ──── WebSocket ────► JARVIS gateway (:8989)
                                                        │
                                                        ├─► Whisper (:9001)
                                                        ├─► Groq / DeepSeek / OpenAI
                                                        └─► Edge TTS
              ◄───────── TTS audio back ◄──────────────────┘
Speakers
```

This is what I use: `.2` is my laptop (mic + speakers), `.3` is a server in the basement (Whisper + LLM + TTS). Audio stays local, compute is remote, latency is fine on a LAN.

## Repository layout

```
jarvis-voice-shell/
├── web/
│   └── jarvis_web.py            FastAPI web UI + WebSocket gateway. Both modes share this.
│                                  1330 lines. Browser-facing HTML/JS embedded.
│
├── jarvis_voice_client.py       Standalone Python client for split-architecture mode.
│                                  733 lines. Pure sounddevice, no browser.
│
├── whisper-server/
│   └── server.py                Faster-Whisper HTTP server. OpenAI-compatible /v1/audio/transcriptions.
│                                  182 lines. Runs on port 9001.
│
├── src/jarvis_voice_shell/      Optional CLI / library mode (push-to-talk, scripted voice).
│   ├── cli.py                   Click CLI — `jarvis-voice run --brain http`
│   ├── bridge.py                Network bridge to a remote JARVIS gateway
│   ├── controller.py            Local controller glue
│   ├── vad.py                   Energy-based VAD (also used by web UI)
│   ├── stt.py / tts.py          STT/TTS engine abstractions
│   ├── recorder.py              Audio capture
│   ├── ptt.py                   Push-to-talk mode
│   ├── openai_voice.py          OpenAI Realtime API integration
│   ├── local_voice.py           Local-only voice mode
│   ├── config.py                Config dataclass
│   ├── latency.py               Latency tracking
│   └── ...
│
├── tests/                       pytest suite (~200 tests). Covers VAD, barge-in, TTS, devices, etc.
│
├── systemd/
│   └── jarvis-voice-client.service   Example systemd unit for the Python client
│
├── docs/plans/                  Design docs and planning notes
│
├── Dockerfile                   Multi-stage: shared base + GPU variant
├── docker-compose.yml           CPU + GPU compose
│
├── requirements.txt             Shared deps
├── requirements-web.txt         Web UI / gateway deps
├── requirements-whisper.txt     Whisper server deps
├── requirements-client.txt      Python client deps
│
├── .env.example                 Configuration template — copy to .env
├── start-all.sh                 Launches Whisper + Web UI together
├── run.sh                       Just the Web UI
├── start-jarvis-linux.sh        Linux launcher for the Python client
│
└── Start-JARVIS*.bat            Windows launchers (Hermes, PTT, OpenAI variants)
```

## Quickstart — single-machine, Docker (easiest)

```bash
git clone https://github.com/Ex8-ca/jarvis-voice-shell.git
cd jarvis-voice-shell

cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_... (or DEEPSEEK_API_KEY / OPENAI_API_KEY)
# Get a free Groq key: https://console.groq.com/keys

docker compose up -d
open http://localhost:8989
```

For GPU acceleration: `docker compose build --build-arg TARGET=gpu` (requires `nvidia-container-toolkit`).

## Quickstart — single-machine, manual

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-web.txt -r requirements-whisper.txt

cp .env.example .env
# Edit .env with your LLM key

# Terminal 1: Whisper STT server
python whisper-server/server.py

# Terminal 2: Web UI + gateway
uvicorn web.jarvis_web:app --host 0.0.0.0 --port 8989

# Browser
open http://localhost:8989
```

## Quickstart — split architecture

### On the server (Whisper + LLM + TTS)

```bash
pip install -r requirements-web.txt -r requirements-whisper.txt
cp .env.example .env
# Set GROQ_API_KEY=***

python whisper-server/server.py &
uvicorn web.jarvis_web:app --host 0.0.0.0 --port 8989
```

The Web UI on `:8989` is also the gateway — clients connect to it via WebSocket.

### On the desktop (mic + speakers)

```bash
git clone https://github.com/Ex8-ca/jarvis-voice-shell.git
cd jarvis-voice-shell
pip install -r requirements-client.txt

# Point at the server
export JARVIS_WS_HOST=192.168.1.50
export JARVIS_WS_PORT=8989

python jarvis_voice_client.py
```

For Linux: `systemd/jarvis-voice-client.service` is a drop-in unit. Set `JARVIS_WS_HOST` in the unit's `Environment=` and enable it.

## LLM provider priority

The gateway picks the first LLM with a key set in `.env`:

| Priority | Provider | Env var | Speed | Cost |
|----------|----------|---------|-------|------|
| 1 | **Groq** | `GROQ_API_KEY` | ~150ms first token | Free tier available |
| 2 | **DeepSeek** | `DEEPSEEK_API_KEY` | ~500ms first token | Pay-per-token, cheap |
| 3 | **OpenAI** | `OPENAI_API_KEY` | ~600ms first token | Most expensive |
| 4 | **Local** | `LOCAL_LLM_URL` | Depends on hardware | Free |
| 5 | **Hermes / OpenAI-compatible** | `HERMES_URL` | Depends | Depends |

For most users, **Groq with `llama-3.1-8b-instant`** is the sweet spot. Free, fast, good enough for voice.

## Latency budget

Typical end-to-end (you-stop-talking → first response audio byte):

| Component | Time |
|-----------|------|
| VAD end-silence | 315ms |
| Whisper STT | ~400ms |
| Filler phrase playback | ~600ms (overlaps with LLM) |
| LLM first token (Groq) | ~150ms |
| LLM full response | ~800ms |
| TTS first chunk | ~400ms |
| **Perceived latency** | **~1.0s** (filler masks LLM) |

The filler phrase ("One sec...") starts playing the moment Whisper returns, before the LLM has even started — so the user perceives ~1s total instead of ~2s.

## Barge-in (interrupting the AI)

The AI's TTS audio bleeds from speakers into the mic. We need to detect when **the user** is talking, not the AI's bleed.

**The hard mute:** the mic is dropped entirely while the AI's audio is in the playback queue. This prevents the AI's audio from being re-transcribed as a new turn. Mic unmutes only when the speaker's queue has fully drained.

**The detection:** the mic still computes RMS on every frame (even when muted), and the barge-in watcher tracks a 2-second rolling baseline of the typical mic level during AI playback. When the current RMS exceeds `max(baseline × 2.5, 800)` sustained for 200ms, barge-in fires. Adapts to your speaker volume automatically — louder speakers → higher threshold.

When barge-in fires, the client:
1. Sends a `barge_in` text message to the gateway
2. Stops local TTS playback immediately (`speaker.stop_immediately()`)
3. Flushes the MP3 buffer

The server:
1. Cancels the in-flight LLM/TTS task
2. Resets VAD to idle
3. Acknowledges with `barge_in_ack`

The new user utterance is then captured normally. The client shows `🚨 Barge-in!` in its log so you can see when it fired.

Tune via `.env`:
```bash
JARVIS_BARGE_IN_BASELINE_RATIO=2.5    # 1.0 = very sensitive, 5.0 = only loud interruptions
JARVIS_BARGE_IN_HOLD_MS=200            # how long speech must sustain to fire
JARVIS_BARGE_IN_BASELINE_WINDOW=100    # 2 seconds of history at 50Hz polling
JARVIS_BARGE_IN_RMS=800                # absolute floor (override baseline if speakers are quiet)
```

## CLI mode (push-to-talk)

For terminal-based voice without a browser:

```bash
pip install -e .
jarvis-voice run --input-mode ptt --brain http
```

`--input-mode vad` for always-on. `--brain http` connects to a JARVIS gateway, `--brain openai` uses OpenAI's Realtime API directly, `--brain local` runs fully offline with a local LLM. See `python -m jarvis_voice_shell.cli --help` for all options.

## Configuration

See `.env.example` for everything. Most important:

```bash
# LLM (one is required)
GROQ_API_KEY=gsk_...

# Voice persona (optional, keep short)
# JARVIS_SYSTEM_PROMPT_FILE=/path/to/your/voice-prompt.txt

# Filler phrases (set empty to disable)
JARVIS_FILLER_PHRASES=One sec...,Checking...,On it...

# Whisper model
WHISPER_MODEL=turbo            # large-v3-turbo (default, best quality)
                                # distil-large-v3 (faster, less accurate)

# TTS voice
JARVIS_TTS_VOICE=en-GB-RyanNeural
```

The gateway also reads `~/.hermes/SOUL.md` and `~/.hermes/USER.md` automatically if they exist (Hermes users).

## Development

```bash
python -m pytest                              # ~200 tests
python -m pytest tests/test_voice_client_bargein.py -v   # barge-in specifically
python -m ruff check .
```

Tests cover VAD, barge-in, audio device resolution, TTS, latency, and config.

## What's NOT here

- **Wake word** — no "Hey JARVIS" keyword spotting. The browser UI is always-on with VAD. If you want a wake word, that's a separate ML model (Porcupine, Vosk) and would be a feature add.
- **Voice activity detection during TTS** — barge-in uses simple RMS, not a neural VAD. Good enough for most environments, but won't catch very quiet interruptions.
- **Conversation history** — each turn is independent. The LLM has no memory of previous turns. Add it by populating the `messages` array in the gateway's LLM call.

## License

MIT
