#!/bin/bash
# Run hermes-voice-client on .2 with the correct audio devices
# Input device 5 = ALC269VC Analog combo jack mic (we unmuted this)
# Input device 7 = pipewire default (BH-M9 Pro BT mic if it's the default source)
# Output device 5 = ALC269VC Analog (wired speaker)
# Output device 7 = pipewire default (BH-M9 Pro BT speaker)
set -e
source ~/hermes-voice-client/venv/bin/activate
export HERMES_WS_HOST="192.168.1.3"
export HERMES_WS_PORT="7979"   # was 8989, now taken by audioforge
export HERMES_INPUT_DEVICE="7"   # pipewire default - follows active BT mic
export HERMES_OUTPUT_DEVICE="7"  # pipewire default - follows active BT speaker
export HERMES_LOG_LEVEL="DEBUG"
exec hermes-voice-client
