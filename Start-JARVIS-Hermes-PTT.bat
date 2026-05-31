@echo off
title JARVIS Voice Shell - Hermes PTT
setlocal

set "BASH_EXE=C:\Program Files\Git\bin\bash.exe"
set "PROJECT_DIR=%~dp0"

if not exist "%BASH_EXE%" (
  echo Git Bash not found at "%BASH_EXE%".
  echo Edit BASH_EXE in this launcher for your system.
  pause
  exit /b 1
)

"%BASH_EXE%" -lc "cd \"$(cygpath -u \"$PROJECT_DIR\")\" && python -m jarvis_voice_shell.cli run --input-mode ptt --brain hermes-cli --stt-engine whisper --stt-model tiny --max-record-seconds 30 --ptt-key '`'; echo; echo Press Enter to close.; read"
