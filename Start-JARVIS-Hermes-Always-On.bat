@echo off
title JARVIS Voice Shell - Hermes Always-On
setlocal

set "BASH_EXE=C:\Program Files\Git\bin\bash.exe"
set "PROJECT_DIR=%~dp0"

if not exist "%BASH_EXE%" (
  echo Git Bash not found at "%BASH_EXE%".
  echo Edit BASH_EXE in this launcher for your system.
  pause
  exit /b 1
)

if "%API_SERVER_KEY%"=="" (
  echo API_SERVER_KEY is not set.
  echo Set a private local gateway key before running this launcher.
  echo Example: set API_SERVER_KEY=your-local-dev-key
  pause
  exit /b 1
)

if "%BRIDGE_MAX_TOKENS%"=="" set "BRIDGE_MAX_TOKENS=512"

echo Starting Hermes gateway...
hermes gateway stop >nul 2>&1
start "Hermes Gateway" /min "%BASH_EXE%" -lc "API_SERVER_ENABLED=true API_SERVER_KEY=\"$API_SERVER_KEY\" API_SERVER_PORT=8642 API_SERVER_HOST=127.0.0.1 hermes gateway run --replace"

set /a TRIES=0
:WAIT_GATEWAY
%SystemRoot%\System32\ping.exe -n 2 127.0.0.1 >nul
curl -sf http://127.0.0.1:8642/health >nul 2>&1
if %errorlevel% equ 0 goto GATEWAY_READY
set /a TRIES+=1
if %TRIES% lss 30 goto WAIT_GATEWAY

echo Hermes gateway failed to start. Check Hermes logs.
pause
exit /b 1

:GATEWAY_READY
echo Hermes gateway online.
"%BASH_EXE%" -lc "cd \"$(cygpath -u \"$PROJECT_DIR\")\" && PYTHONUNBUFFERED=1 API_SERVER_KEY=\"$API_SERVER_KEY\" HERMES_BRIDGE_URL=http://127.0.0.1:8642/v1/chat/completions BRIDGE_MAX_TOKENS=\"$BRIDGE_MAX_TOKENS\" python -u -m jarvis_voice_shell.cli run --input-mode always-on --brain http --sample-rate 16000 --tts-rate +10%% --stt-engine whisper --stt-model tiny --max-record-seconds 8 --vad-threshold 300 --vad-end-silence-ms 700; echo; echo Press Enter to close.; read"
