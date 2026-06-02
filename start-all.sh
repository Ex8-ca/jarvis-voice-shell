#!/bin/bash
# Start the hermes-voice gateway together with the Whisper STT server.
# Whisper is started in the background; the gateway runs in the foreground.
set -e

# Load .env if present
if [ -f .env ]; then
    set -a
    . .env
    set +a
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Use venv if it exists, otherwise the system python
if [ -d venv ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON=python3
fi

# Whisper STT server — use the distil-large-v3 model by default. Distil is
# trained for conversational/short audio, so it hallucinates far less on
# voice commands than the large-v3-turbo default. Override with
# WHISPER_MODEL=large-v3-turbo if you need higher accuracy on long dictation.
WHISPER_MODEL="${WHISPER_MODEL:-Systran/faster-distil-whisper-large-v3}"
WHISPER_PORT="${WHISPER_PORT:-9001}"
WHISPER_SCRIPT="${WHISPER_SCRIPT:-/home/marc/whisper-server/server.py}"

echo "Starting Whisper STT server (model=$WHISPER_MODEL) on :$WHISPER_PORT..."
WHISPER_MODEL="$WHISPER_MODEL" WHISPER_PORT="$WHISPER_PORT" \
    python3 "$WHISPER_SCRIPT" > /tmp/whisper.log 2>&1 &
WHISPER_PID=$!
trap "kill $WHISPER_PID 2>/dev/null" EXIT

# Wait for Whisper to be ready (up to 60s for first load + model download)
for i in {1..60}; do
    if curl -sf "http://127.0.0.1:$WHISPER_PORT/health" > /dev/null 2>&1; then
        echo "Whisper ready."
        break
    fi
    sleep 1
done

# hermes-voice gateway
echo "Starting hermes-voice gateway on :8989..."
exec /home/marc/.hermes/hermes-agent/venv/bin/uvicorn \
    hermes_voice.gateway:app --host 0.0.0.0 --port 8989
