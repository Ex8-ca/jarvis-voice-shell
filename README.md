# JARVIS Voice Shell

A local-first voice shell for [Hermes Agent](https://github.com/nousresearch/hermes-agent). It keeps speech I/O in a small Python app while Hermes remains the reasoning, tools, memory, and profile layer.

> JARVIS is used here as an assistant-style project name. This project is not affiliated with Marvel, Disney, OpenAI, Microsoft, or Nous Research.

## Architecture

```text
Microphone → VAD / PTT → Whisper STT → Hermes Gateway HTTP bridge → Edge TTS → Speakers
                                      ↘ optional OpenAI Realtime voice mode
```

The preferred path is Hermes Gateway HTTP mode:

- `jarvis_voice_shell` handles audio, VAD/PTT, STT, TTS, turn cancellation, and speech sanitisation.
- Hermes Gateway exposes an OpenAI-compatible local HTTP endpoint.
- Hermes keeps the agent behaviour: tools, memory, skills, model routing, and profile config.

## Features

- Typed, push-to-talk, and always-on VAD input modes
- Local Whisper STT via `faster-whisper`
- Edge TTS voice output, defaulting to `en-GB-RyanNeural`
- Hermes Gateway HTTP bridge with bearer auth and SSE streaming support
- Optional slower `hermes chat` subprocess bridge for compatibility
- Optional OpenAI Realtime mode as an ears/mouth layer
- Turn controller and TTS cancellation for barge-in experiments
- Test suite covering bridge, config, audio device selection, recorder, TTS, STT, and VAD logic

## Install

```bash
python -m pip install -e ".[dev,stt]"
```

For audio device access, you may also need platform-specific PortAudio/PyAudio support. The code falls back to `sounddevice` where possible.

## Configure

Copy the example environment file and fill in only what you need:

```bash
cp .env.example .env
```

Important variables:

```bash
API_SERVER_KEY=your-local-gateway-key
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
HERMES_BRIDGE_URL=http://127.0.0.1:8642/v1/chat/completions
BRIDGE_MAX_TOKENS=512
```

`API_SERVER_KEY` is required for Hermes Gateway HTTP mode. Use any private local value, but do not commit it.

## Linux Setup

On Linux (Ubuntu/Pop!_OS tested), use the provided setup and launcher scripts:

```bash
# Install system dependencies, create venv, install Python packages
bash setup-linux.sh

# Configure
cp .env.example .env
# Edit .env with your Hermes API key and bridge URL

# Launch (always-on VAD mode)
bash start-jarvis-linux.sh
```

If the wrong microphone is selected, list devices and pass an explicit index:
```bash
python -m jarvis_voice_shell.cli list-devices
bash start-jarvis-linux.sh <device_index>
```

## Remote Hermes Gateway

JARVIS can talk to a Hermes Agent running on a different machine. Set `HERMES_BRIDGE_URL` in `.env` to point at the remote endpoint:

```bash
HERMES_BRIDGE_URL=http://192.168.1.3:6789/v1/chat/completions
API_SERVER_KEY=your-gateway-key
```

All audio processing (VAD, Whisper STT, Edge TTS, playback) remains local — only the LLM call goes over the network.

## Start Hermes Gateway

In one terminal:

```bash
API_SERVER_ENABLED=true \
API_SERVER_KEY="$API_SERVER_KEY" \
API_SERVER_HOST=127.0.0.1 \
API_SERVER_PORT=8642 \
hermes gateway run --replace
```

Check it:

```bash
curl -sf http://127.0.0.1:8642/health
curl -sf -H "Authorization: Bearer ${API_SERVER_KEY}" http://127.0.0.1:8642/v1/models
```

## Run

Typed mode:

```bash
python -m jarvis_voice_shell.cli run --input-mode typed --brain http
```

Push-to-talk mode:

```bash
python -m jarvis_voice_shell.cli run --input-mode ptt --brain http --stt-engine whisper --stt-model tiny
```

Always-on VAD mode:

```bash
python -m jarvis_voice_shell.cli run \
  --input-mode always-on \
  --brain http \
  --sample-rate 16000 \
  --tts-rate +10% \
  --stt-engine whisper \
  --stt-model tiny \
  --max-record-seconds 8 \
  --vad-threshold 300 \
  --vad-end-silence-ms 700
```

If device auto-detection selects the wrong microphone or speaker, first inspect devices using Python:

```bash
python - <<'PY'
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], 'in=', d['max_input_channels'], 'out=', d['max_output_channels'], 'sr=', d['default_samplerate'])
PY
```

Then pass `--input-device <index>` and/or `--output-device <index>`.

## Windows launchers

The included `.bat` files are public-safe examples. They use the script directory rather than a hardcoded user path and require `API_SERVER_KEY` to be set before running Gateway mode.

If your Git Bash path differs from `C:\Program Files\Git\bin\bash.exe`, edit the launcher locally. Do not commit personal paths or device indices.

## Development

```bash
python -m pytest
python -m ruff check .
```

Current expected test command:

```bash
python -m pytest
```

## Security / public sharing notes

This repository intentionally excludes:

- `.env` and other secret files
- audio recordings and generated TTS files
- cache directories and bytecode
- local Hermes runtime state
- personal Windows paths and machine-specific audio device IDs

Before publishing, run:

```bash
python -m pytest
python scripts/public_safety_scan.py
```

## Known limitations

- Edge TTS uses a network service.
- Always-on VAD thresholds are microphone-dependent.
- True full-duplex voice remains experimental; chunked/streaming TTS is the next major latency improvement.
- Hermes Gateway must be running for `--brain http` mode.
