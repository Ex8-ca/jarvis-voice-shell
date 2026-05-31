@echo off
title JARVIS - Hermes Brain + OpenAI Realtime Voice Layer
setlocal

set "BASH_EXE=C:\Program Files\Git\bin\bash.exe"
set "PROJECT_DIR=%~dp0"

if not exist "%BASH_EXE%" (
  echo Git Bash not found at "%BASH_EXE%".
  echo Edit BASH_EXE in this launcher for your system.
  pause
  exit /b 1
)

if "%OPENAI_API_KEY%"=="" (
  echo OPENAI_API_KEY is not set.
  echo Set it before using OpenAI Realtime voice mode.
  pause
  exit /b 1
)

"%BASH_EXE%" -lc "cd \"$(cygpath -u \"$PROJECT_DIR\")\" && python -m jarvis_voice_shell.cli openai-voice --brain hermes-cli --sample-rate 24000 --model gpt-realtime-mini --voice marin; echo; echo Press Enter to close.; read"
