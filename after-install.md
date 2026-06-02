# Hermes Voice — Installed ✓

Your Hermes agent now has voice capabilities.

## What was installed

- **Voice server** — FastAPI app with VAD, Whisper STT, LLM, and Edge TTS
- **Browser UI** — tap-to-talk voice interface at port 8989
- **Hermes tools** — `hermes_voice_status` tool and `/hermes-voice` slash command

## Quick start

1. **Open the voice UI** in your browser:
   ```
   http://localhost:8989/
   ```
   (Or your Tailscale URL if accessing remotely.)

2. **Grant microphone access** when prompted.

3. **Tap 🎤 and speak** — the server auto-detects when you stop talking.

## Server management

From any Hermes session (CLI, Telegram, Discord, etc.):

```
/hermes-voice status    — check if the server is running
/hermes-voice start     — start the server
/hermes-voice stop      — stop the server
/hermes-voice restart   — restart the server
```

## Required environment variables

Make sure these are set in `~/.hermes/.env`:

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | LLM for voice responses |
| `WHISPER_URL` | Local Whisper STT server (default: `http://127.0.0.1:9001/v1/audio/transcriptions`) |

If the server doesn't start, check that the Whisper server is running:
```bash
cd ~/whisper-server && python3 server.py
```

## Customization

- **Voice persona**: Edit `~/.hermes/VOICE.md` to change how the assistant speaks.
- **Assistant name**: Set `HERMES_NAME` in `.env` (default: "Hermes").
- **Port**: Set `HERMES_VOICE_PORT` in `.env` (default: 8989).
