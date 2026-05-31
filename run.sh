#!/bin/bash
# Launch JARVIS web server with DeepSeek API key from Hermes .env
source /home/marc/.hermes/.env 2>/dev/null
cd /home/marc/jarvis-voice-shell
exec ./venv/bin/python -m uvicorn web.jarvis_web:app --host 0.0.0.0 --port 8989
