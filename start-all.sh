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

# Whisper STT server — use the original (non-distil) turbo model by default.
# Distil hallucinates less but is slower on CPU. The non-distil turbo with
# int8 quantization is faster and was the model that worked reliably before
# the nvidia driver crash. Override with WHISPER_MODEL=... if needed.
#
# On GPU (RTX 5080 / Blackwell, sm_120), use float16 + ctranslate2>=4.7.2:
# - ctranslate2 4.7.1 falls back to a slow path on Blackwell (~1.4x realtime)
# - ctranslate2 4.7.2+ has sm_120 kernels (~0.13x realtime, ~30x faster)
# - GPU detection: faster-whisper uses CUDA if available, else CPU
# - Override: WHISPER_COMPUTE_TYPE=int8 for CPU, float16 for GPU
WHISPER_MODEL="${WHISPER_MODEL:-mobiuslabsgmbh/faster-whisper-large-v3-turbo}"

# Auto-pick compute type: float16 on GPU, int8 on CPU
if [ -z "$WHISPER_COMPUTE_TYPE" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        WHISPER_COMPUTE_TYPE="float16"
        echo "GPU detected — using float16 (requires ctranslate2>=4.7.2 for sm_120)"
    else
        WHISPER_COMPUTE_TYPE="int8"
        echo "No GPU — using int8"
    fi
fi

WHISPER_BEAM_SIZE="${WHISPER_BEAM_SIZE:-1}"
WHISPER_PORT="${WHISPER_PORT:-9001}"
WHISPER_SCRIPT="${WHISPER_SCRIPT:-/home/marc/whisper-server/server.py}"

echo "Starting Whisper STT server (model=$WHISPER_MODEL, compute=$WHISPER_COMPUTE_TYPE, beam=$WHISPER_BEAM_SIZE) on :$WHISPER_PORT..."
WHISPER_MODEL="$WHISPER_MODEL" WHISPER_PORT="$WHISPER_PORT" \
    WHISPER_COMPUTE_TYPE="$WHISPER_COMPUTE_TYPE" WHISPER_BEAM_SIZE="$WHISPER_BEAM_SIZE" \
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

# hermes-voice gateway — port 7979 (audioforge owns 8989)
HERMES_VOICE_PORT="${HERMES_VOICE_PORT:-7979}"
echo "Starting hermes-voice gateway on :$HERMES_VOICE_PORT..."
exec /home/marc/.hermes/hermes-agent/venv/bin/uvicorn \
    hermes_voice.gateway:app --host 0.0.0.0 --port "$HERMES_VOICE_PORT"
