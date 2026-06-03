#!/bin/bash
# Start the hermes-voice gateway together with the Whisper STT server.
# Whisper is started in the background; the gateway runs in the foreground.
#
# Hardware tier auto-detection: this script picks the best Whisper
# configuration for your hardware:
#   Tier 1 (CPU only)        — int8, no interim STT
#   Tier 2 (modern GPU)     — float16, interim STT on
#   Tier 3 (enthusiast GPU) — float16, multiple sessions possible
#
# Override: WHISPER_COMPUTE_TYPE=int8|float16|float32
#           WHISPER_BEAM_SIZE=1|5
#           HERMES_INTERIM_STT=0|1
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

# ---------------------------------------------------------------------------
# Hardware tier detection
# ---------------------------------------------------------------------------
# Tier 1: CPU only — int8 quantization, no interim STT (CPU can't keep up
#          with parallel calls)
# Tier 2: Modern NVIDIA GPU with 8GB+ VRAM, or Apple Silicon — float16,
#          interim STT enabled
# Tier 3: 16GB+ NVIDIA GPU — same as Tier 2 but with more headroom
# ---------------------------------------------------------------------------

detect_tier() {
    # 1. Try NVIDIA first
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        local vram_mib
        vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        # Round to nearest GB. A 16303 MiB card (RTX 5080) rounds to 16 GB,
        # not 15 GB. Integer division alone would misclassify 16GB cards
        # that report just under 16384 MiB.
        local vram_gb=$(( (vram_mib + 512) / 1024 ))
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ "$vram_gb" -ge 16 ] 2>/dev/null; then
            echo "TIER3:$vram_gb:$gpu_name"
        elif [ "$vram_gb" -ge 8 ] 2>/dev/null; then
            echo "TIER2:$vram_gb:$gpu_name"
        elif [ "$vram_gb" -ge 4 ] 2>/dev/null; then
            echo "TIER2:$vram_gb:$gpu_name"  # small GPU but better than CPU
        else
            echo "TIER1:0:no-nvidia"
        fi
        return
    fi

    # 2. Try Apple Silicon (M1/M2/M3/M4)
    if [ "$(uname)" = "Darwin" ] && sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "apple"; then
        local mem_gb
        mem_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))
        if [ "$mem_gb" -ge 8 ] 2>/dev/null; then
            echo "TIER2:$mem_gb:apple-silicon"
        else
            echo "TIER1:0:apple-silicon-low-ram"
        fi
        return
    fi

    # 3. CPU only
    echo "TIER1:0:cpu"
}

# Print a friendly banner showing the detected tier
TIER_INFO=$(detect_tier)
TIER=$(echo "$TIER_INFO" | cut -d: -f1)
DETAIL=$(echo "$TIER_INFO" | cut -d: -f2-)

# Write a tier marker for the gateway's /health endpoint to read.
# Format: just one word on the first line (gpu | apple | cpu).
case "$TIER" in
    TIER3|TIER2) echo "gpu" > /tmp/hermes-voice-tier ;;
    TIER1)
        if echo "$DETAIL" | grep -qi "apple"; then
            echo "apple" > /tmp/hermes-voice-tier
        else
            echo "cpu" > /tmp/hermes-voice-tier
        fi
        ;;
esac
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
case "$TIER" in
    TIER3)
        echo "  Hardware tier: 3 (enthusiast GPU)"
        echo "  Detected:      $DETAIL GB VRAM"
        echo "  Config:        float16, beam=1, interim STT on"
        ;;
    TIER2)
        echo "  Hardware tier: 2 (modern desktop — recommended)"
        echo "  Detected:      $DETAIL"
        echo "  Config:        float16, beam=1, interim STT on"
        ;;
    TIER1)
        echo "  Hardware tier: 1 (CPU only — works on anything)"
        echo "  Detected:      $DETAIL"
        echo "  Config:        int8, beam=1, interim STT off"
        echo "  Note:          ~9s per 1s of audio. A modern GPU would be 25x faster."
        ;;
esac
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Whisper model selection. The non-distil turbo is what we use because:
#  - It has the best quality / speed tradeoff
#  - Distil was tried but Marc found it unreliable for short voice commands
#  - Smaller models (base, small) are noticeably worse on accented English
WHISPER_MODEL="${WHISPER_MODEL:-mobiuslabsgmbh/faster-whisper-large-v3-turbo}"

# Auto-pick compute type and interim STT based on tier (if not overridden)
if [ -z "$WHISPER_COMPUTE_TYPE" ]; then
    case "$TIER" in
        TIER1) WHISPER_COMPUTE_TYPE="int8" ;;
        TIER2|TIER3) WHISPER_COMPUTE_TYPE="float16" ;;
    esac
fi

if [ -z "$HERMES_INTERIM_STT" ]; then
    case "$TIER" in
        TIER1) HERMES_INTERIM_STT="0" ;;
        TIER2|TIER3) HERMES_INTERIM_STT="1" ;;
    esac
fi

WHISPER_BEAM_SIZE="${WHISPER_BEAM_SIZE:-1}"
WHISPER_PORT="${WHISPER_PORT:-9001}"
WHISPER_SCRIPT="${WHISPER_SCRIPT:-/home/marc/whisper-server/server.py}"

# CTranslate2 sanity check for NVIDIA users
if [ "$TIER" != "TIER1" ]; then
    CT_VER=$("$PYTHON" -c "import ctranslate2; print(ctranslate2.__version__)" 2>/dev/null || echo "0.0.0")
    CT_MAJOR=$(echo "$CT_VER" | cut -d. -f1)
    CT_MINOR=$(echo "$CT_VER" | cut -d. -f2)
    if [ "$CT_MAJOR" -lt 4 ] 2>/dev/null || { [ "$CT_MAJOR" -eq 4 ] && [ "$CT_MINOR" -lt 7 ]; } 2>/dev/null; then
        echo ""
        echo "  ⚠  ctranslate2 $CT_VER is too old for modern GPUs."
        echo "     On RTX 40/50 series (Blackwell/Ada), you need ctranslate2>=4.7.2"
        echo "     or Whisper will fall back to a 10x slower generic path."
        echo "     Fix: pip install --upgrade ctranslate2"
        echo ""
    fi
fi

export HERMES_INTERIM_STT
echo "Starting Whisper STT server (model=$WHISPER_MODEL, compute=$WHISPER_COMPUTE_TYPE, beam=$WHISPER_BEAM_SIZE, interim=$HERMES_INTERIM_STT) on :$WHISPER_PORT..."
WHISPER_MODEL="$WHISPER_MODEL" WHISPER_PORT="$WHISPER_PORT" \
    WHISPER_COMPUTE_TYPE="$WHISPER_COMPUTE_TYPE" WHISPER_BEAM_SIZE="$WHISPER_BEAM_SIZE" \
    "$PYTHON" "$WHISPER_SCRIPT" > /tmp/whisper.log 2>&1 &
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
