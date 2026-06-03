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

Mic + speakers on your desktop; STT + LLM + TTS on a server. Useful when the desktop is underpowered or you want a "voice satellite" setup. The Python client (`hermes_voice.client`) does the audio I/O locally; everything heavy runs remotely.

```
Desktop / voice satellite                      Server / GPU box
─────────────────────                          ────────────────
Microphone ──► Python client ──── WebSocket ────► Hermes gateway (:7979)
                                                        │
                                                        ├─► Whisper (:9001)
                                                        ├─► Groq / DeepSeek / OpenAI
                                                        └─► Edge TTS
              ◄───────── TTS audio back ◄──────────────────┘
Speakers
```

Common split setup: a small/quiet box on your desk (laptop, mini-PC, Raspberry Pi) handles mic + speakers, and a more powerful machine on the LAN (or over Tailscale) runs Whisper + LLM + TTS. Audio stays local; compute is remote; latency is fine on a LAN.

## Repository layout

```
hermes-voice/
├── web/
│   └── jarvis_web.py            FastAPI web UI + WebSocket gateway. Both modes share this.
│                                  1330 lines. Browser-facing HTML/JS embedded.
│
├── hermes_voice/
│   ├── client.py                Python voice client for split-architecture mode.
│   │                              750+ lines. Pure sounddevice, no browser. Sends
│   │                              raw PCM over WebSocket to the gateway.
│   ├── gateway.py               FastAPI WebSocket gateway + REST /health endpoint.
│   ├── vad.py                   Energy-based VAD
│   ├── llm.py / tts.py / memory.py / persona.py / naming.py
│   └── skills_registry.py / tools/    Plugin-internal skill registration
│
├── web/jarvis_web.py            Legacy single-machine FastAPI web UI (browser-based).
│                                  Optional — use the plugin's gateway unless you
│                                  want a browser UI on a remote box.
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
│   └── jarvis-voice-client.service   Drop-in systemd --user unit for the Python client
│
├── docs/plans/                  Design docs and planning notes
│
├── Dockerfile                   Multi-stage: shared base + GPU variant
├── docker-compose.yml           CPU + GPU compose
│
├── requirements.txt             Shared deps
├── requirements-web.txt         Web UI / gateway deps
├── requirements-whisper.txt     Whisper server deps
├── requirements-client.txt      Python client deps (sounddevice, numpy, websockets, miniaudio, python-dotenv)
│
├── .env.example                 Configuration template — copy to .env
├── start-all.sh                 Launches Whisper + gateway together (server side)
├── run.sh                       Just the gateway (server side)
├── bootstrap.sh                 Server-side install: venv + Whisper deps + ctranslate2
├── bootstrap-client.sh          Client-side install: venv + 5 small packages, no Whisper
```

## Prerequisites

**One Python venv. One LLM API key. Optional GPU.** That's the whole list.

| What | Why | Notes |
|---|---|---|
| **Python 3.10+** | Runs the gateway + Whisper server | `python3 --version` to check |
| **LLM API key** | Real-time responses | Free: [Groq](https://console.groq.com/keys) · Cheap: [DeepSeek](https://platform.deepseek.com/) · Any OpenAI-compatible URL |
| **Whisper** (local) | Speech-to-text without sending audio to the cloud | Runs on your CPU or GPU; bundled `whisper-server/server.py` |
| **Edge TTS** | Text-to-speech | Free, runs in the cloud (Microsoft), no key needed |
| **GPU (optional)** | Faster Whisper (~30x realtime vs ~1x) | Any NVIDIA 8GB+ works. Apple Silicon works too. CPU is fine, just slower. |

> **Don't run `pip install` manually.**
> - **Server side** (Whisper + gateway + LLM + TTS): run `./bootstrap.sh`
> - **Client side** (desktop mic + speakers, split-architecture): run `./bootstrap-client.sh`
>
> Each script picks the right deps, sets up the venv, and offers to start the
> relevant services. See the [Quickstart](#quickstart) below.

### System packages (Linux only)

The Python wheels cover almost everything, but a few C libraries need apt/dnf/pacman:

```bash
# Debian/Ubuntu
sudo apt install -y python3-dev portaudio19-dev libasound2-dev ffmpeg

# Fedora
sudo dnf install -y python3-devel portaudio-devel alsa-lib-devel ffmpeg

# Arch
sudo pacman -S python portaudio alsa-lib ffmpeg
```

`portaudio` is for the desktop mic client (split-architecture mode). `ffmpeg` is for Whisper to read non-WAV audio. The web UI doesn't need either.

## Quickstart — pick your path

| Path | Best for | Time |
|---|---|---|
| **[A. Docker](#quickstart--single-machine-docker-easiest)** | Don't want to touch Python at all | 2 min |
| **[B. Plugin (recommended for Hermes users)](#quickstart--install-as-a-hermes-plugin)** | You already run Hermes Agent | 3 min |
| **[C. Manual (no Docker, no plugin)](#quickstart--single-machine-manual)** | Want full control, no abstraction | 5 min |
| **[D. Split architecture](#quickstart--split-architecture)** | Mic on desktop, voice on a server | 10 min |

---

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

**Three commands. The plugin manages the server; you just chat.**

```bash
# 1. Install the plugin
hermes plugins install Ex8-ca/hermes-voice

# 2. Run the one-time dependency installer (creates venv, installs Whisper, etc.)
#    The first time, this downloads ~1.5GB of Whisper model weights.
cd ~/.hermes/plugins/hermes-voice && ./bootstrap.sh

# 3. Start the voice server (any chat — Telegram, Discord, CLI)
#    In a Hermes session, send:
/hermes-voice start
```

Then open **http://localhost:8989** in your browser. Grant mic permission, talk.

### What each step does

| Step | What it actually does | Skip if… |
|---|---|---|
| `hermes plugins install` | `git clone` into `~/.hermes/plugins/hermes-voice/`, generates `.env` from `.env.example` | already installed? run `hermes plugins update hermes-voice` |
| `./bootstrap.sh` | Creates Python venv, installs `faster-whisper` + `edge-tts` + `ctranslate2`, sets up `whisper-server`, writes `WHISPER_URL` to `.env`, downloads the default model | you already have a working venv + Whisper on a port |
| `/hermes-voice start` | Spawns the gateway in the background on port 7979 (or `HERMES_VOICE_PORT` if you set one) | you set up a systemd service to start it on boot |

> **Default port is 7979.** If another service is already using it, set
> `HERMES_VOICE_PORT=<some-free-port>` in `.env` before starting. Run
> `ss -tln | grep 7979` (or `lsof -i :7979`) to see what's there.

### Verify it worked

```bash
curl http://localhost:7979/health
```

Should return:
```json
{"status": "ok", "uptime_s": 12.3, "whisper": "ok", "whisper_url": "...", "port": 7979, "tier": "gpu", "version": "0.1.0"}
```

Or from any chat: `/hermes-voice status` returns a multi-line version of the same.

### Common install problems

| Error in the logs | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'faster_whisper'` | `cd ~/.hermes/plugins/hermes-voice && ./bootstrap.sh` |
| `Whisper not reachable at http://127.0.0.1:9001` | Same — `bootstrap.sh` starts it for you |
| `port 7979 already in use` | Set `HERMES_VOICE_PORT=7978` (or any free port) in `.env` |
| `ImportError: libcudart.so` (GPU) | `pip install --upgrade ctranslate2` (the plugin does this) |
| `/hermes-voice` doesn't appear in chat | Restart the gateway: `hermes gateway restart` |

### Updating

```bash
cd ~/.hermes/plugins/hermes-voice
git pull                       # (rare — usually `hermes plugins update` does this)
./bootstrap.sh                 # picks up any new requirements
/hermes-voice restart          # reload gateway with new code
```

### What the plugin auto-registers

- **Tool:** `hermes_voice_status` — agent can check if voice is healthy before sending a voice reply
- **Slash command:** `/hermes-voice start|stop|restart|status [port]` — works in every connected chat
- **Auto-start:** gateway spawns when Hermes loads the plugin, no extra action needed
- **Health endpoint:** `GET /health` returns JSON for monitoring scripts

## Quickstart — single-machine, manual

If you're not using Docker and not using the plugin, do this:

```bash
git clone https://github.com/Ex8-ca/hermes-voice.git
cd hermes-voice

# The one-shot installer handles Python deps, venv, Whisper, and .env
./bootstrap.sh

# (Optional) download the Whisper model now so first run is fast
./bootstrap.sh --download-model

# Edit .env and set your LLM key (Groq is free, see link in Prerequisites)
# Then start everything:
./start-all.sh

# Browser
open http://localhost:8989
```

`bootstrap.sh` is idempotent — run it again any time to update deps. Use `./bootstrap.sh --with-client` if you also want the Python desktop mic client.

## Quickstart — split architecture

You run the **mic + speakers** on one machine (often a laptop or a Raspberry Pi
sitting on your desk), and the **Whisper + LLM + TTS** on another (a beefier
box with a GPU). The two talk over WebSocket.

### On the server (Whisper + LLM + TTS)

```bash
git clone https://github.com/Ex8-ca/hermes-voice.git
cd hermes-voice
./bootstrap.sh              # venv + Whisper + deps
# Edit .env and set GROQ_API_KEY=***
./start-all.sh              # starts Whisper + gateway on :7979 (or HERMES_VOICE_PORT)
```

The Web UI on `HERMES_VOICE_PORT` (default 7979) is also the gateway — clients
connect to it via WebSocket. Verify it's up:

```bash
curl http://localhost:7979/health
# → {"status":"ok","uptime_s":...,"whisper":"ok","tier":"gpu",...}
```

### On the desktop (mic + speakers)

The desktop only needs Python + 4 small packages (sounddevice, numpy,
websockets, miniaudio, python-dotenv). No Whisper, no GPU.

**1. Install system packages** (required for `sounddevice` to compile):

```bash
# Debian / Ubuntu / Pop!_OS
sudo apt install -y libportaudio2 portaudio19-dev libasound2-dev

# Fedora
sudo dnf install -y portaudio portaudio-devel alsa-lib-devel

# Arch
sudo pacman -S portaudio alsa-lib
```

**2. Clone and bootstrap the client side:**

```bash
git clone https://github.com/Ex8-ca/hermes-voice.git
cd hermes-voice
./bootstrap-client.sh 192.168.1.50 7979
# (host and port are the server's IP and HERMES_VOICE_PORT)
# Prompts for GROQ_API_KEY — same key the server uses.
```

`bootstrap-client.sh` creates a venv, installs the 5 client packages, writes
`.env` with `chmod 600`, and verifies the import. Idempotent — safe to re-run.

**3. Run the client:**

```bash
# From the repo dir
./venv/bin/python -m hermes_voice.client
# Or, if you've activated the venv:
source venv/bin/activate
python -m hermes_voice.client
```

You should see:

```
[INFO] hermes-voice-client: hermes-voice client → ws://192.168.1.50:7979/ws
[INFO] hermes-voice-client: WebSocket connected
[INFO] hermes-voice-client: mic: frame SENT (RMS=0, muted=False)
```

Speak into the mic. `RMS=` should jump from 0 to 200-2000+ while you talk,
and you'll hear the AI's response through your default speaker.

**4. Pick the right mic.** If the wrong device is being used (e.g. your
webcam mic instead of your headset), find devices and force the index:

```bash
./venv/bin/python -c "import sounddevice; print(sounddevice.query_devices())"
# Look at the 'name' and 'index' columns.

# Force a specific input device:
HERMES_VOICE_INPUT_DEVICE=5 ./venv/bin/python -m hermes_voice.client
# (Or add `HERMES_VOICE_INPUT_DEVICE=5` to your .env to make it permanent.)
```

### Environment variables (client side)

| Var | Default | What it does |
|---|---|---|
| `HERMES_VOICE_WS_HOST` | `127.0.0.1` | Gateway hostname or IP |
| `HERMES_VOICE_WS_PORT` | `8989` | Gateway WebSocket port (use `7979` for the plugin's default) |
| `HERMES_VOICE_INPUT_DEVICE` | system default | sounddevice input device index |
| `HERMES_VOICE_OUTPUT_DEVICE` | system default | sounddevice output device index |
| `GROQ_API_KEY` | — | Same key the gateway uses; needed for some LLM paths |

All of these can go in `~/.hermes-voice/.env` and the client will read them
automatically. Real env vars still take precedence.

### Auto-start on boot (Linux)

A drop-in systemd --user unit lives at `systemd/jarvis-voice-client.service`.
Adjust the `WorkingDirectory`, `ExecStart`, and any `Environment=` lines, then:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/jarvis-voice-client.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jarvis-voice-client
loginctl enable-linger marc   # keep it running after logout
```

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

### Tier 3 — enthusiast / server

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

Reference measurements on an RTX 5080 (16GB VRAM, ctranslate2 4.7.2):

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

## Troubleshooting

### Client: `ModuleNotFoundError: No module named 'sounddevice'`

The venv has numpy, websockets, and python-dotenv, but `sounddevice` and
`miniaudio` failed to install. **Most common cause:** missing PortAudio headers.

```bash
sudo apt install -y libportaudio2 portaudio19-dev libasound2-dev   # Debian/Ubuntu
sudo dnf install -y portaudio portaudio-devel alsa-lib-devel       # Fedora
sudo pacman -S portaudio alsa-lib                                 # Arch
```

Then re-run `./bootstrap-client.sh` — it's idempotent. The two packages will
compile against the headers and install cleanly.

### Client: `hermes-voice client → ws://127.0.0.1:8989/ws` (ignoring your .env)

The client connected to localhost instead of your gateway's IP. Two causes:

1. **You're using the system Python** (`python3 -m hermes_voice.client`)
   without activating the venv. `python-dotenv` is only in the venv, so the
   `.env` file is silently ignored. Use `./venv/bin/python -m hermes_voice.client`
   instead — no `source venv/bin/activate` needed.

2. **The `.env` file is missing or has placeholder values.** Check that
   `HERMES_VOICE_WS_HOST=<your gateway IP>` and `HERMES_VOICE_WS_PORT=7979`
   are set, and the IP is reachable (`nc -vz <host> 7979` should say `succeeded`).

### Client: `mic: frame SENT (RMS=0, muted=False)` even while I'm talking

The mic is opening and frames are flowing, but they're silent. The wrong
audio device is selected. PulseAudio/PipeWire's "default" often resolves to
a webcam mic or a Bluetooth headset that's currently disconnected.

```bash
./venv/bin/python -c "import sounddevice; print(sounddevice.query_devices())"
```

Find the right mic's index, then:

```bash
HERMES_VOICE_INPUT_DEVICE=5 ./venv/bin/python -m hermes_voice.client
```

Add `HERMES_VOICE_INPUT_DEVICE=5` to `.env` to make it permanent.

**Bluetooth headset gotcha:** if you disconnect/reconnect the headset, the
default device index can shift. Disconnect and reconnect the headset, or
re-run the device-list command to find the new index.

### Server: `port 7979 already in use`

Another process is already bound to 7979. Pick a different port and update
both the server and the client to use it.

```bash
# See what's on 7979
ss -tlnp | grep 7979
lsof -i :7979

# Pick a free port (7978, 8000, 8080 — anything not in use)
echo "HERMES_VOICE_PORT=7978" >> .env

# On the desktop client side, point to the same port
echo "HERMES_VOICE_WS_PORT=7978" >> .env

# Restart the gateway
/hermes-voice restart
```

If you're not actively using the conflicting service, you can free 7979 by
stopping it — but the safer default is to leave it alone and pick a different
port for hermes-voice.

### Server: `/health` says `whisper: down`

The gateway is up but can't reach the Whisper server. Usually means
`start-all.sh` wasn't used (Whisper isn't running) or Whisper crashed.

```bash
# Check if Whisper is on 9001
curl -s http://127.0.0.1:9001/v1/models

# Or just restart everything cleanly
./start-all.sh
```

### Server: STT takes 1.5x realtime on a modern GPU

Your `ctranslate2` is too old. For RTX 40-series, 50-series, or anything
Blackwell/Ada/Hopper, you need `ctranslate2>=4.7.2`:

```bash
./venv/bin/python -m pip install --upgrade 'ctranslate2>=4.7.2'
# Then restart start-all.sh
```

### Client connects, server logs "WebSocket voice connected", but no audio round-trips

The WebSocket is open, but the client isn't actually capturing mic audio.
Check the client log for `mic: frame SENT (RMS=N, muted=False)` — if `RMS=0`
and never moves, the mic stream opened successfully but on a silent device.
See the "wrong device" troubleshooting above.

If `RMS` moves but the server still gets no audio, check the client log for
barge-in events firing repeatedly — the mic may be getting muted because the
AI's TTS audio is bleeding into it. Lower your speaker volume, or use
headphones.

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
