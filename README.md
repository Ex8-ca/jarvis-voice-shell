# hermes-voice

Real-time voice pipeline for Hermes Agent. Streaming STT → streaming LLM → streaming TTS, with filler phrases that mask latency and a wake-word-free always-on mic. Runs in two modes: single-machine (mic + STT + LLM + TTS in one box, web UI in your browser) or split (mic on one machine, STT/LLM/TTS on another over WebSocket).

> Hermes is the assistant name used here. The legacy "JARVIS" naming survives in some file paths and the `jarvis-voice` CLI entry point (for the push-to-talk mode); the web UI, plugin, and docs use Hermes.

## What you get

- **Always-on mic** with energy-based VAD — no push-to-talk, no wake word needed for the web UI
- **Streaming Whisper STT** on the server (Faster-Whisper, `large-v3-turbo` by default)
- **Streaming LLM** — Groq (~150ms first token), DeepSeek, OpenAI, local (Ollama/vLLM), or any OpenAI-compatible proxy. First available key wins.
- **Filler phrases** ("One sec...", "Checking...") play the instant you stop talking, so the AI's pause feels like thinking, not lag
- **Edge TTS** streams MP3 chunks back to the client as they synthesize (no waiting for the full audio)
- **Barge-in** — interrupt the AI mid-response by talking over it
- **Hardware tier auto-detection** — CPU, modern GPU, and Apple Silicon paths are picked automatically. See [Hardware tiers](#hardware-tiers).
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
hermes-voice/
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
git clone https://github.com/Ex8-ca/hermes-voice.git
cd hermes-voice

cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_... (or DEEPSEEK_API_KEY / OPENAI_API_KEY)
# Get a free Groq key: https://console.groq.com/keys

docker compose up -d
open http://localhost:8989
```

For GPU acceleration: `docker compose build --build-arg TARGET=gpu` (requires `nvidia-container-toolkit`).

## Quickstart — install as a Hermes plugin

```bash
hermes plugins install Ex8-ca/hermes-voice
hermes plugins enable hermes-voice
hermes gateway restart

open http://localhost:8989
```

The plugin auto-starts the voice server on load. Use `/hermes-voice start|stop|restart|status` from any Hermes session.

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
git clone https://github.com/Ex8-ca/hermes-voice.git
cd hermes-voice
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

End-to-end latency from "you stop talking" to "first response audio byte":

| Component | CPU tier (no GPU) | GPU tier (8GB+ VRAM) |
|-----------|-------------------|----------------------|
| VAD end-silence | 1.5s | 1.5s |
| Whisper STT (per 1s of audio) | ~9s | **~0.4s** |
| LLM first token (Groq) | ~150ms | ~150ms |
| LLM full response | ~800ms | ~800ms |
| TTS first chunk | ~400ms | ~400ms |
| **Total per 1s of speech** | **~12s** | **~3s** |
| **Perceived (filler masks LLM)** | **~10s** | **~2.5s** |

**Why the gap is so big:** Whisper is the bottleneck. On CPU it does roughly 1x realtime (1s of audio takes 1s to process). On a modern GPU with `ctranslate2>=4.7.2` it does 30x realtime (1s of audio takes 0.03s to process). Everything else in the pipeline is fast on any hardware.

**The filler phrase** ("One sec...") starts playing the moment Whisper returns, before the LLM has even started — so you perceive ~1s of latency from the moment you stop talking, with the AI's pause feeling like thinking, not lag.

## Hardware tiers

Hermes Voice auto-detects your hardware and picks the best configuration. The `start-all.sh` script does this automatically — no manual setup needed.

### Tier 1 — works on anything (CPU only)

**What you need:** Any computer, no GPU required.

| | |
|---|---|
| Whisper model | `large-v3-turbo` |
| Compute type | `int8` |
| Beam size | 1 |
| Time per 1s of audio | ~9s |
| Interim STT | disabled (CPU can't keep up with parallel calls) |
| Feel | "Alexa in 2014" — slow but works |

**Use this if:** you're on a laptop without a discrete GPU, or you're on a Raspberry Pi, or you're just trying things out.

### Tier 2 — modern desktop / Apple Silicon (recommended)

**What you need:** Discrete NVIDIA GPU with 8GB+ VRAM (RTX 3060, 4060, or better) OR an Apple Silicon Mac (M1/M2/M3/M4) with 8GB+ unified memory.

| | |
|---|---|
| Whisper model | `large-v3-turbo` |
| Compute type | `float16` |
| Beam size | 1 |
| Time per 1s of audio | ~0.4s |
| Interim STT | enabled (sub-second partial transcripts as you talk) |
| Feel | "ChatGPT Voice" — instant and natural |

**Use this if:** you have a gaming PC from the last 3-4 years, a Mac with Apple Silicon, or a workstation with an NVIDIA card. This covers most desktop users in 2026.

### Tier 3 — enthusiast / server (our setup)

**What you need:** 16GB+ NVIDIA GPU (RTX 4080/5080, A4000, etc.) and a beefy CPU. This is the setup you'd run on a home server or workstation.

Same software as Tier 2, just more headroom. Useful if you're running multiple voice sessions, batch-processing audio, or planning to add local LLMs later (Llama 8B on a 16GB GPU is a future possibility).

### Auto-detection

The `start-all.sh` script picks your tier automatically:

```bash
./start-all.sh
# GPU detected — using float16 (requires ctranslate2>=4.7.2 for sm_120)
# Starting Whisper STT server (model=...turbo, compute=float16, beam=1) on :9001...
# Whisper ready.
# Starting hermes-voice gateway on :7979...
```

Manual override if auto-detection is wrong:

```bash
# Force CPU mode
WHISPER_COMPUTE_TYPE=int8 ./start-all.sh

# Force GPU mode (will fail loudly if CUDA isn't actually available)
WHISPER_COMPUTE_TYPE=float16 ./start-all.sh
```

### Speed reference

Real measurements on our hardware (RTX 5080, 16GB VRAM, ctranslate2 4.7.2):

| Audio length | Cold (first run) | Warm (subsequent) |
|---|---|---|
| 1 second | ~0.5s | ~0.05s |
| 11 seconds (JFK sample) | 1.4s | **0.35s** |
| 30 seconds | ~3.5s | ~1s |

On CPU (no GPU, int8, beam=1) the same audio takes 9-15s regardless of length.

### Why ctranslate2 version matters

If you have an NVIDIA GPU from 2024 or later (RTX 40-series, RTX 50-series, anything Blackwell / Ada Lovelace / Hopper), you need `ctranslate2>=4.7.2`. Earlier versions fall back to a slow generic path because they don't have the new GPU kernels.

If you see Whisper taking 1.5x realtime on a modern GPU, your `ctranslate2` is too old:

```bash
pip install --upgrade ctranslate2
```

The `requirements-whisper.txt` already pins this. If you built Whisper from a system package manager instead of pip, check the version it ships.

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
