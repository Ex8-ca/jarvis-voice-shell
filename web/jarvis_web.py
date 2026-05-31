"""
JARVIS Voice Web Interface
Uses browser audio streaming → EnergyVAD → local Faster Whisper STT → DeepSeek direct → Edge TTS.
No cloud speech services — entirely local pipeline.
"""

import logging
import os
from pathlib import Path
import tempfile

import httpx
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-web")

# ── Energy VAD (reused from Ex8-ca/jarvis-voice-shell) ─────────────────

import math
from enum import Enum

class VADState(str, Enum):
    IDLE = "idle"
    PRIMED = "primed"
    SPEAKING = "speaking"

def rms_int16(frame: bytes) -> int:
    if not frame:
        return 0
    sample_count = len(frame) // 2
    if sample_count <= 0:
        return 0
    total = 0
    for i in range(0, sample_count * 2, 2):
        sample = int.from_bytes(frame[i:i + 2], "little", signed=True)
        total += sample * sample
    return int(math.sqrt(total / sample_count))

class EnergyVAD:
    def __init__(
        self,
        energy_threshold: int = 500,
        start_frames: int = 3,
        end_silence_frames: int = 11,
        pre_roll_frames: int = 5,
        sample_width: int = 2,  # bytes per sample
        frame_ms: int = 63,     # ms per chunk from browser
    ):
        self.energy_threshold = int(energy_threshold)
        self.start_frames = max(1, int(start_frames))
        self.end_silence_frames = max(1, int(end_silence_frames))
        self.pre_roll_frames = max(0, int(pre_roll_frames))
        self.sample_width = int(sample_width)
        # frames per ms → per chunk
        self.samples_per_frame = 16000 * frame_ms // 1000  # 1008 samples @ 63ms
        self.state = VADState.IDLE
        self._pre_roll: list[bytes] = []
        self._primed: list[bytes] = []
        self._segment: list[bytes] = []
        self._loud_count = 0
        self._quiet_count = 0

    def _frame_rms(self, frame: bytes) -> int:
        """Return RMS amplitude of a PCM frame."""
        return rms_int16(frame)

    def process(self, frame: bytes) -> bytes | None:
        loud = self._frame_rms(frame) >= self.energy_threshold

        if self.state == VADState.IDLE:
            if loud:
                self.state = VADState.PRIMED
                self._primed = [frame]
                self._loud_count = 1
            else:
                # Keep recent frames for pre-roll
                if self.pre_roll_frames > 0:
                    self._pre_roll.append(frame)
                    # trim pre-roll to maxlen
                    over = len(self._pre_roll) - self.pre_roll_frames
                    if over > 0:
                        self._pre_roll = self._pre_roll[over:]
            return None

        if self.state == VADState.PRIMED:
            if loud:
                self._primed.append(frame)
                self._loud_count += 1
                if self._loud_count >= self.start_frames:
                    self.state = VADState.SPEAKING
                    self._segment = list(self._pre_roll) + self._primed
                    self._quiet_count = 0
                    self._pre_roll = []
                    self._primed = []
            else:
                # quiet frame in PRIMED — reset
                self._pre_roll.extend(self._primed)
                self._pre_roll.append(frame)
                # trim pre-roll
                over = len(self._pre_roll) - self.pre_roll_frames
                if over > 0:
                    self._pre_roll = self._pre_roll[over:]
                self._primed = []
                self._loud_count = 0
                self.state = VADState.IDLE
            return None

        # SPEAKING state
        self._segment.append(frame)
        if loud:
            self._quiet_count = 0
            return None
        self._quiet_count += 1
        if self._quiet_count >= self.end_silence_frames:
            segment = b"".join(self._segment)
            self.reset()
            return segment
        return None

    def reset(self) -> None:
        self.state = VADState.IDLE
        self._pre_roll = []
        self._primed = []
        self._segment = []
        self._loud_count = 0
        self._quiet_count = 0

app = FastAPI(title="JARVIS Voice Web")

# Load .env file for DEEPSEEK_API_KEY (pitfall fix)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Hermes gateway config (for full-context requests)
HERMES_URL = "http://192.168.1.3:6789/v1/chat/completions"
HERMES_API_KEY = "chillygeek6789"
HERMES_MODEL = "deepseek-chat"

# Direct DeepSeek config (for fast voice turns — 5K tokens vs 43K)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Lightweight system prompt for voice turns (SOUL.md + USER.md only)
VOICE_SYSTEM_PROMPT = None  # loaded lazily

def _load_voice_prompt():
    global VOICE_SYSTEM_PROMPT
    if VOICE_SYSTEM_PROMPT is not None:
        return VOICE_SYSTEM_PROMPT
    
    parts = []
    for fname in [Path.home() / ".hermes" / "SOUL.md", Path.home() / ".hermes" / "USER.md"]:
        if fname.exists():
            parts.append(fname.read_text())
    
    VOICE_SYSTEM_PROMPT = (
        "\n\n".join(parts) + 
        "\n\n---\nYou are JARVIS, Marc's voice assistant. Keep responses SHORT — under 30 words. "
        "Conversational, direct, no filler. You are speaking aloud, not typing."
    )
    return VOICE_SYSTEM_PROMPT

# Local Whisper STT server
WHISPER_URL = "http://192.168.1.3:9001/v1/audio/transcriptions"

# TTS cache directory
CACHE_DIR = Path.home() / ".cache" / "jarvis-voice-shell" / "tts_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS Voice</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .container { max-width: 600px; width: 100%; text-align: center; }
        h1 {
            font-size: 2rem; margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            margin: 20px 0; padding: 15px; border-radius: 12px;
            background: #1a1a2e; border: 1px solid #333;
        }
        .status.listening { border-color: #00ff88; box-shadow: 0 0 20px rgba(0,255,136,0.3); }
        .status.thinking { border-color: #ffd700; box-shadow: 0 0 20px rgba(255,215,0,0.3); }
        .status.speaking { border-color: #00d4ff; box-shadow: 0 0 20px rgba(0,212,255,0.3); }
        .btn {
            display: inline-block; padding: 20px 40px; margin: 20px 0;
            font-size: 1.5rem; border: none; border-radius: 50%;
            cursor: pointer; transition: all 0.3s ease;
            width: 120px; height: 120px;
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: #000; font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,212,255,0.4);
        }
        .btn:active { transform: scale(0.95); }
        .btn.listening {
            background: linear-gradient(135deg, #ff0055, #ff6600);
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255,0,85,0.4); }
            50% { box-shadow: 0 0 40px rgba(255,0,85,0.8); }
        }
        .conversation {
            margin-top: 20px; text-align: left;
            max-height: 60vh; overflow-y: auto;
        }
        .message {
            margin: 10px 0; padding: 12px 16px;
            border-radius: 12px; max-width: 85%;
        }
        .message.user {
            background: #1a1a2e; margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .message.jarvis {
            background: #2a2a3e; border-bottom-left-radius: 4px;
        }
        .message .label { font-size: 0.8rem; color: #888; margin-bottom: 4px; }
        .latency { font-size: 0.75rem; color: #666; margin-top: 4px; }
        .cursor {
            animation: blink 1s step-end infinite;
            color: #00ff88;
        }
        @keyframes blink {
            50% { opacity: 0; }
        }
        #interim-display {
            transition: color 0.3s;
        }
        .timer {
            font-size: 2rem; color: #00ff88; margin-top: 10px;
            font-variant-numeric: tabular-nums;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>JARVIS <span style="color:#00ff88;font-size:14px;vertical-align:middle;margin-left:8px;font-weight:normal">v2.3</span></h1>
        <p style="color: #666; margin-bottom: 20px;">Tap 🎤 and speak — auto-detects when you stop</p>

        <div class="status" id="status">Ready</div>
        <div id="vad-indicator" style="height:4px;width:100%;max-width:400px;margin:-10px auto 10px;border-radius:2px;background:#222;overflow:hidden;">
            <div id="vad-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#00d4ff,#00ff88);transition:width 0.1s ease;"></div>
        </div>
        <div id="interim-display" style="min-height:24px;max-width:500px;margin:0 auto 15px;color:#888;font-style:italic;font-size:0.95rem;transition:color 0.3s;"></div>

        <button class="btn" id="mic-btn" onclick="toggleVoice()">🎤</button>
        <div class="timer" id="timer" style="display:none"></div>

        <div style="margin: 15px 0;">
            <input type="text" id="text-input" placeholder="Or type your message..."
                   style="width: 70%; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a2e; color: #e0e0e0; font-size: 1rem;">
            <button onclick="sendText()"
                    style="padding: 12px 20px; border-radius: 8px; border: 1px solid #333; background: #00d4ff; color: #000; font-weight: bold; cursor: pointer;">Send</button>
        </div>

        <p style="font-size: 0.8rem; color: #666;">🎙 VAD → Whisper STT → DeepSeek → Edge TTS</p>

        <div class="conversation" id="conversation"></div>

        <button onclick="clearChat()" style="margin-top: 20px; padding: 8px 16px; background: #333; color: #888; border: 1px solid #444; border-radius: 8px; cursor: pointer;">Clear Chat</button>
    </div>

    <script>
        let audioContext = null;
        let processor = null;
        let stream = null;
        let isListening = false;
        let recordingStart = 0;
        let timerInterval = null;
        let vadEnergy = 0;
        let autoListen = true; // Keep mic live between turns

        function showStatus(msg, cls) {
            document.getElementById('status').textContent = msg;
            document.getElementById('status').className = 'status' + (cls ? ' ' + cls : '');
        }

        // ── Toggle recording ──────────────────────────────────────────
        async function toggleVoice() {
            if (isListening) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        async function startRecording() {
            try {
                // Stop wake word listener before using mic
                stopWakeWordListener();
                showStatus('Requesting mic...');
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isListening) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    // Float32 [-1, 1] → Int16 PCM little-endian
                    const pcm = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    const buf = pcm.buffer;
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(buf);
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                recordingStart = Date.now();
                isListening = true;

                document.getElementById('mic-btn').classList.add('listening');
                document.getElementById('mic-btn').innerHTML = '⏹';
                showStatus('🎙 Listening... speak naturally', 'listening');

                document.getElementById('timer').style.display = 'block';
                timerInterval = setInterval(() => {
                    const elapsed = ((Date.now() - recordingStart) / 1000).toFixed(1);
                    document.getElementById('timer').textContent = elapsed + 's';
                }, 100);

            } catch (err) {
                console.error('Mic error:', err);
                if (err.name === 'NotAllowedError') showStatus('⚠ Mic permission denied');
                else if (err.name === 'NotFoundError') showStatus('⚠ No microphone found');
                else showStatus('⚠ Mic error: ' + err.message);
                resetUI();
            }
        }

        function stopRecording() {
            isListening = false;
            clearInterval(timerInterval);
            document.getElementById('timer').style.display = 'none';
            document.getElementById('vad-bar').style.width = '0%';

            if (processor) { try { processor.disconnect(); } catch {} processor = null; }
            if (audioContext) { try { audioContext.close(); } catch {} audioContext = null; }
            if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        }

        function resetUI() {
            stopRecording();
            document.getElementById('mic-btn').classList.remove('listening');
            document.getElementById('mic-btn').innerHTML = '🎤';
        }

        // ── VAD energy visualizer ────────────────────────────────────
        function updateVadBar(state, energy) {
            const bar = document.getElementById('vad-bar');
            if (state === 'idle') {
                bar.style.width = energy ? Math.min(100, energy) + '%' : '0%';
                bar.style.background = 'linear-gradient(90deg,#00d4ff,#00ff88)';
            } else if (state === 'primed') {
                bar.style.width = '60%';
                bar.style.background = 'linear-gradient(90deg,#ffd700,#ff9500)';
            } else if (state === 'speaking') {
                bar.style.width = '100%';
                bar.style.background = 'linear-gradient(90deg,#ff0055,#ff6600)';
            } else if (state === 'processing') {
                bar.style.width = '100%';
                bar.style.background = 'linear-gradient(90deg,#00d4ff,#00ff88)';
            }
        }

        // ── WebSocket voice pipeline ────────────────────────────────────
        let ws = null;
        let audioChunksReceived = [];
        let currentAudio = null;
        let processing = false;

        function connectWebSocket() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => console.log('WS connected');
            ws.onclose = () => { console.log('WS closed — reconnecting in 2s'); setTimeout(connectWebSocket, 2000); };
            ws.onerror = (e) => console.error('WS error', e);

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);
                    switch (msg.type) {
                        case 'vad_state':
                            updateVadBar(msg.state, msg.energy || 0);
                            if (msg.state === 'processing') {
                                processing = true;
                                // Clear any stale streaming response from cancelled turn
                                clearStreamingResponse();
                                showStatus('Processing...', 'thinking');
                            } else if (msg.state === 'speaking') {
                                showStatus('Speaking...', 'speaking');
                            } else if (msg.state === 'idle' && !processing) {
                                showStatus('🎙 Listening... speak naturally', 'listening');
                            }
                            break;

                        case 'interim_transcript':
                            // Show partial transcript while user is still talking
                            updateInterim(msg.text);
                            break;

                        case 'transcript':
                            showStatus(`Heard: "${msg.text}"`, 'listening');
                            // Clear interim + any stale streaming response
                            clearInterim();
                            clearStreamingResponse();
                            addMessage(msg.text, 'user');
                            break;

                        case 'token':
                            // Streaming LLM token — append to current response bubble
                            appendStreamingToken(msg.text);
                            break;

                        case 'response_complete':
                            // LLM streaming done — finalize the response bubble
                            finalizeStreamingResponse(msg);
                            break;

                        case 'speaking':
                            showStatus('Speaking...', 'speaking');
                            break;

                        case 'done':
                            processing = false;
                            if (audioChunksReceived.length > 0) {
                                playAudioChunks();
                            }
                            showStatus('🎙 Listening... speak naturally', 'listening');
                            // Keep mic live for next turn
                            if (!isListening) { startRecording().catch(() => {}); }
                            break;

                        case 'error':
                            processing = false;
                            showStatus('⚠ ' + msg.text);
                            clearInterim();
                            if (!isListening) { startRecording().catch(() => {}); }
                            break;
                    }
                } else {
                    // Binary audio chunk
                    audioChunksReceived.push(new Uint8Array(event.data));
                }
            };
        }

        function playAudioChunks() {
            if (currentAudio) { currentAudio.pause(); currentAudio = null; }
            const blob = new Blob(audioChunksReceived, { type: 'audio/mpeg' });
            const url = URL.createObjectURL(blob);
            currentAudio = new Audio(url);
            currentAudio.play();
            currentAudio.onended = () => { URL.revokeObjectURL(url); currentAudio = null; };
        }

        connectWebSocket();

        // ── Wake word detection (hey Jarvis) ──────────────────────────
        // Uses energy-based detection — RMS spike over threshold triggers wake
        const WW_RMS_THRESHOLD = 1500; // higher than speech — only loud "hey Jarvis" triggers
        const WW_CONFIRM_FRAMES = 3;   // must stay above threshold for 3 consecutive frames
        let ww_rms_history = [];

        async function stopWakeWordListener() {
            if (window.wwProcessor) { try { window.wwProcessor.disconnect(); } catch {} window.wwProcessor = null; }
            if (window.wwAudioCtx) { try { window.wwAudioCtx.close(); } catch {} window.wwAudioCtx = null; }
            if (window.wwStream) { window.wwStream.getTracks().forEach(t => t.stop()); window.wwStream = null; }
        }

        async function wakeWordListen() {
            if (!autoListen) return;
            try {
                await stopWakeWordListener();
                window.wwStream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
                });
                window.wwAudioCtx = new AudioContext({ sampleRate: 16000 });
                const source = window.wwAudioCtx.createMediaStreamSource(window.wwStream);
                window.wwProcessor = window.wwAudioCtx.createScriptProcessor(4096, 1, 1);

                window.wwProcessor.onaudioprocess = (e) => {
                    const inputData = e.inputBuffer.getChannelData(0);
                    let rms = 0;
                    for (let i = 0; i < inputData.length; i++) rms += inputData[i] * inputData[i];
                    rms = Math.sqrt(rms / inputData.length) * 32768;

                    ww_rms_history.push(rms);
                    if (ww_rms_history.length > 10) ww_rms_history.shift();

                    const avgRms = ww_rms_history.reduce((a, b) => a + b, 0) / ww_rms_history.length;

                    if (avgRms > WW_RMS_THRESHOLD) {
                        ww_rms_history = [];
                        stopWakeWordListener();
                        // Wake word detected — start full recording
                        startRecording().catch(() => {});
                    }
                };

                source.connect(window.wwProcessor);
                window.wwProcessor.connect(window.wwAudioCtx.destination);
            } catch {}
        }

        // Start wake word listener on page load
        wakeWordListen();

        // ── Chat with Hermes ──────────────────────────────────────────
        let conversation = JSON.parse(localStorage.getItem('jarvis_convo') || '[]');
        renderConversation();

        async function processText(text, extra = {}) {
            showStatus('Thinking...', 'thinking');
            addMessage(text, 'user');

            try {
                const response = await fetch(`/chat?text=${encodeURIComponent(text)}`);
                const data = await response.json();

                addMessage(data.response, 'jarvis', {
                    stt_ms: extra.stt_ms || 0,
                    bridge_ms: data.bridge_ms,
                    tts_ms: data.tts_ms,
                    total_ms: data.total_ms,
                });

                if (data.audio_url) {
                    showStatus('Speaking...', 'speaking');
                    const audio = new Audio(data.audio_url);
                    audio.play();
                    audio.onended = () => { showStatus('Ready'); resetUI(); };
                } else {
                    showStatus('Ready');
                    resetUI();
                }

            } catch (err) {
                showStatus('Error: ' + err.message);
                console.error(err);
                resetUI();
            }
        }

        function addMessage(text, role, latency = null) {
            conversation.push({ text, role, latency, time: Date.now() });
            localStorage.setItem('jarvis_convo', JSON.stringify(conversation));
            renderConversation();
        }

        function renderConversation() {
            const container = document.getElementById('conversation');
            container.innerHTML = conversation.map(msg => {
                let l = '';
                if (msg.latency && msg.role === 'jarvis') {
                    const parts = [];
                    if (msg.latency.stt_ms) parts.push(`STT ${msg.latency.stt_ms}ms`);
                    parts.push(`LLM ${msg.latency.bridge_ms?.toFixed(0) || '?'}ms`);
                    parts.push(`TTS ${msg.latency.tts_ms?.toFixed(0) || '?'}ms`);
                    l = `<div class="latency">⏱ ${parts.join(' → ')}</div>`;
                }
                return `<div class="message ${msg.role}">
                    <div class="label">${msg.role === 'user' ? 'You' : 'JARVIS'}</div>
                    <div>${msg.text}</div>${l}
                </div>`;
            }).join('');
            container.scrollTop = container.scrollHeight;
        }

        function clearChat() {
            conversation = [];
            localStorage.removeItem('jarvis_convo');
            document.getElementById('conversation').innerHTML = '';
        }

        // ── Interim transcript display ───────────────────────────────
        function updateInterim(text) {
            const el = document.getElementById('interim-display');
            if (el) {
                el.textContent = '🎙 ' + text;
                el.style.color = '#aaa';
            }
        }

        function clearInterim() {
            const el = document.getElementById('interim-display');
            if (el) {
                el.textContent = '';
                el.style.color = '#888';
            }
        }

        // ── Streaming LLM response ────────────────────────────────────
        let streamingMsgId = null;
        let streamingText = '';

        function appendStreamingToken(token) {
            const container = document.getElementById('conversation');
            streamingText += token;

            if (!streamingMsgId) {
                // Create new streaming bubble
                streamingMsgId = 'streaming-' + Date.now();
                const div = document.createElement('div');
                div.id = streamingMsgId;
                div.className = 'message jarvis';
                div.innerHTML = '<div class="label">JARVIS</div><div class="streaming-text">' + escapeHtml(streamingText) + '<span class="cursor">▊</span></div>';
                container.appendChild(div);
            } else {
                // Update existing bubble
                const div = document.getElementById(streamingMsgId);
                if (div) {
                    div.querySelector('.streaming-text').innerHTML = escapeHtml(streamingText) + '<span class="cursor">▊</span>';
                }
            }
            container.scrollTop = container.scrollHeight;
        }

        function finalizeStreamingResponse(msg) {
            const container = document.getElementById('conversation');
            if (streamingMsgId) {
                const div = document.getElementById(streamingMsgId);
                if (div) {
                    // Remove cursor, add latency
                    const sttPart = msg.stt_ms ? 'STT ' + msg.stt_ms + 'ms → ' : '';
                    const latencyHtml = '<div class="latency">⏱ ' + sttPart + 'LLM ' + (msg.llm_ms?.toFixed(0) || '?') + 'ms</div>';
                    div.querySelector('.streaming-text').innerHTML = escapeHtml(streamingText);
                    div.innerHTML += latencyHtml;
                }
            } else {
                // No streaming happened (response was cached/instant)
                addMessage(msg.text, 'jarvis', { bridge_ms: msg.llm_ms || 0 });
            }

            // Save to conversation history
            conversation.push({ text: streamingText || msg.text, role: 'jarvis', latency: { bridge_ms: msg.llm_ms || 0 }, time: Date.now() });
            localStorage.setItem('jarvis_convo', JSON.stringify(conversation));

            // Reset streaming state
            streamingMsgId = null;
            streamingText = '';
            audioChunksReceived = [];
        }

        function clearStreamingResponse() {
            if (streamingMsgId) {
                const div = document.getElementById(streamingMsgId);
                if (div) {
                    div.innerHTML = '<div class="label">JARVIS</div><div style="color:#666;font-style:italic">cancelled</div>';
                }
                streamingMsgId = null;
                streamingText = '';
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function sendText() {
            const input = document.getElementById('text-input');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';
            await processText(text);
        }

        document.getElementById('text-input')?.addEventListener('keypress', e => {
            if (e.key === 'Enter') sendText();
        });
    </script>
</body>
</html>""",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── STT proxy: browser audio → local Whisper ─────────────────────────

@app.post("/stt")
async def stt_transcribe(file: UploadFile = File(...)):
    """Proxy audio to local Faster Whisper server, return transcription."""
    audio_bytes = await file.read()

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                WHISPER_URL,
                files={"file": (file.filename or "audio.webm", audio_bytes, file.content_type or "audio/webm")},
                data={"model": "whisper-1", "response_format": "json"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"text": "", "error": "Whisper server not ready — model still loading. Try again in a minute."}
    except Exception as e:
        logger.exception("STT proxy error")
        return {"text": "", "error": str(e)}


# ── WebSocket voice pipeline with VAD ─────────────────────────────────

@app.websocket("/ws")
async def voice_websocket(ws: WebSocket):
    """Real-time voice pipeline over WebSocket with streaming STT + LLM.
    
    Browser streams raw PCM → EnergyVAD detects speech.
    WHILE user is speaking: interim Whisper STT every ~500ms.
    When speech ends: final STT → streaming LLM tokens → TTS audio.
    New speech cancels any in-progress response.
    
    Protocol (server → browser):
      {type: 'vad_state', state: 'idle'|'primed'|'speaking'|'processing'}
      {type: 'interim_transcript', text: '...'}          — partial STT during speech
      {type: 'transcript', text: '...', stt_ms: N}       — final STT
      {type: 'token', text: '...'}                        — streaming LLM token
      {type: 'response_complete', text: '...', llm_ms: N} — LLM done, full text
      {type: 'speaking'}                                   — TTS starting
      binary audio/mp3 chunks                              — TTS streaming
      {type: 'done'}                                       — turn complete
      {type: 'error', text: '...'}
    """
    import time, edge_tts, hashlib, struct, asyncio, json as _json, wave

    await ws.accept()
    logger.info("WebSocket voice connected (streaming mode)")

    FRAME_MS = 63
    SAMPLES_PER_FRAME = 16000 * FRAME_MS // 1000  # 1008

    vad = EnergyVAD(
        energy_threshold=300,
        start_frames=3,
        end_silence_frames=5,   # 5 × 63ms = 315ms silence (was 11 = 693ms)
        pre_roll_frames=5,
    )

    processing = False
    current_task: asyncio.Task | None = None

    # Interim STT tracking
    speech_buffer: list[bytes] = []
    interim_seq = 0
    last_interim_send = 0.0
    INTERIM_INTERVAL = 0.5  # seconds between interim Whisper calls

    async def send_interim_stt(audio_data: bytes, seq: int):
        """Send partial audio to Whisper for interim transcription."""
        if len(audio_data) < 16000:  # min 1 second of audio
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            w = wave.open(tmp_path, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio_data)
            w.close()

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                with open(tmp_path, "rb") as f:
                    stt_resp = await client.post(
                        WHISPER_URL,
                        files={"file": ("audio.wav", f, "audio/wav")},
                        data={"model": "whisper-1", "response_format": "json"},
                    )
                stt_resp.raise_for_status()
                stt_data = stt_resp.json()
            text = stt_data.get("text", "").strip()
            if text and seq == interim_seq:  # only if still current
                try:
                    await ws.send_json({"type": "interim_transcript", "text": text})
                except WebSocketDisconnect:
                    pass
        except Exception:
            pass  # interim failures are non-critical
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def receive_loop():
        """Continuously receive PCM chunks from browser and feed VAD."""
        nonlocal speech_buffer, interim_seq, last_interim_send, processing, current_task

        try:
            while True:
                data = await ws.receive_bytes()
                if len(data) < SAMPLES_PER_FRAME * 2:
                    continue

                old_state = vad.state.value
                segment = vad.process(data)
                new_state = vad.state.value

                if new_state != old_state:
                    await ws.send_json({"type": "vad_state", "state": new_state})

                # During speech: send periodic interim STT
                if vad.state == VADState.SPEAKING:
                    speech_buffer.append(data)
                    now = time.monotonic()
                    if now - last_interim_send > INTERIM_INTERVAL:
                        last_interim_send = now
                        interim_seq += 1
                        audio_snapshot = b"".join(speech_buffer)
                        asyncio.create_task(send_interim_stt(audio_snapshot, interim_seq))

                if segment:
                    # Speech segment complete — cancel any in-progress processing
                    if current_task and not current_task.done():
                        current_task.cancel()
                        logger.info("Cancelled previous processing task — new speech detected")

                    # Invalidate pending interim requests
                    interim_seq += 1
                    # Use accumulated speech buffer if we have it, otherwise VAD segment
                    final_audio = b"".join(speech_buffer) if speech_buffer else segment
                    speech_buffer = []

                    processing = True
                    await ws.send_json({"type": "vad_state", "state": "processing"})
                    current_task = asyncio.create_task(process_segment(final_audio))

                if vad.state == VADState.IDLE:
                    speech_buffer = []

        except WebSocketDisconnect:
            pass

    async def process_segment(audio_data: bytes):
        """Run one speech segment: STT → streaming LLM → TTS."""
        nonlocal processing

        t0 = time.perf_counter()
        audio_len_s = len(audio_data) // 2 // 16000
        audio_rms_val = rms_int16(audio_data) if audio_data else 0
        logger.info(f"process_segment: {len(audio_data)} bytes (~{audio_len_s:.1f}s), RMS={audio_rms_val}")

        if audio_rms_val < 50:
            await ws.send_json({"type": "error", "text": "Audio level too low"})
            processing = False
            return

        # ── Step 1: STT via Whisper ────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            w = wave.open(tmp_path, "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio_data)
            w.close()

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                with open(tmp_path, "rb") as f:
                    stt_resp = await client.post(
                        WHISPER_URL,
                        files={"file": ("audio.wav", f, "audio/wav")},
                        data={"model": "whisper-1", "response_format": "json"},
                    )
                stt_resp.raise_for_status()
                stt_data = stt_resp.json()
            transcript = stt_data.get("text", "").strip()
        except httpx.ConnectError:
            await ws.send_json({"type": "error", "text": "Whisper not ready — model loading"})
            processing = False
            return
        except Exception:
            logger.exception("STT error")
            processing = False
            return
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if not transcript:
            await ws.send_json({"type": "error", "text": "No speech detected"})
            processing = False
            return

        stt_ms = (time.perf_counter() - t0) * 1000
        await ws.send_json({"type": "transcript", "text": transcript, "stt_ms": stt_ms})

        # ── Step 2: Streaming LLM (DeepSeek) ──────────────────────
        llm_start = time.perf_counter()
        system_prompt = _load_voice_prompt()
        auth_key = DEEPSEEK_API_KEY or HERMES_API_KEY
        api_url = DEEPSEEK_URL if DEEPSEEK_API_KEY else HERMES_URL
        model = DEEPSEEK_MODEL if DEEPSEEK_API_KEY else HERMES_MODEL
        max_tok = 120 if DEEPSEEK_API_KEY else 300

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}]
        if not DEEPSEEK_API_KEY:
            messages = [{"role": "user", "content": transcript}]

        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", api_url,
                    json={"model": model, "messages": messages, "max_tokens": max_tok, "stream": True},
                    headers={"Authorization": f"Bearer {auth_key}", "Content-Type": "application/json"},
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = _json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                                    await ws.send_json({"type": "token", "text": content})
                            except Exception:
                                continue
        except Exception as e:
            logger.exception("LLM streaming error")
            if not full_response:
                await ws.send_json({"type": "error", "text": f"LLM error: {e}"})
                processing = False
                return

        llm_ms = (time.perf_counter() - llm_start) * 1000

        if not full_response:
            await ws.send_json({"type": "error", "text": "Empty response"})
            processing = False
            return

        await ws.send_json({"type": "response_complete", "text": full_response, "llm_ms": llm_ms})

        # ── Step 3: TTS streaming ─────────────────────────────────
        await ws.send_json({"type": "speaking"})

        text_hash = hashlib.sha256(full_response.encode()).hexdigest()[:16]
        cache_path = CACHE_DIR / f"tts_{text_hash}.mp3"

        if cache_path.exists():
            tts_data = cache_path.read_bytes()
            await ws.send_bytes(tts_data)
        else:
            communicate = edge_tts.Communicate(text=full_response, voice="en-GB-RyanNeural", rate="+0%")
            tts_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    tts_chunks.append(chunk["data"])
                    await ws.send_bytes(chunk["data"])
            cache_path.write_bytes(b"".join(tts_chunks))

        await ws.send_json({"type": "done"})
        processing = False
        vad.reset()
        try:
            await ws.send_json({"type": "vad_state", "state": "idle"})
        except WebSocketDisconnect:
            pass

    # Start receive loop
    receive_task = asyncio.create_task(receive_loop())
    try:
        await receive_task
    except asyncio.CancelledError:
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass


# ── Chat endpoints ────────────────────────────────────────────────────

@app.get("/chat")
async def chat_get(text: str):
    return await _process_chat(text)


@app.post("/chat")
async def chat_post(text: str):
    return await _process_chat(text)


async def _process_chat(text: str):
    """Voice-optimized: DeepSeek direct (SOUL.md + USER.md only) → Edge TTS."""
    import time
    import edge_tts
    import hashlib

    total_start = time.perf_counter()

    # Step 1: Call DeepSeek directly with lightweight system prompt
    bridge_start = time.perf_counter()
    system_prompt = _load_voice_prompt()
    
    auth_key = DEEPSEEK_API_KEY or HERMES_API_KEY
    api_url = DEEPSEEK_URL if DEEPSEEK_API_KEY else HERMES_URL
    model = DEEPSEEK_MODEL if DEEPSEEK_API_KEY else HERMES_MODEL
    max_tok = 150 if DEEPSEEK_API_KEY else 300
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    if not DEEPSEEK_API_KEY:
        messages = [{"role": "user", "content": text}]  # Hermes injects its own system prompt

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            api_url,
            json={"model": model, "messages": messages, "max_tokens": max_tok, "stream": False},
            headers={"Authorization": "Bearer " + auth_key, "Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
    bridge_ms = (time.perf_counter() - bridge_start) * 1000

    # Step 2: Generate TTS using edge-tts
    tts_start = time.perf_counter()
    text_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"tts_{text_hash}.mp3"

    if not cache_path.exists():
        communicate = edge_tts.Communicate(
            text=response_text,
            voice="en-GB-RyanNeural",
            rate="+0%",
        )
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        audio_data = b"".join(chunks)
        cache_path.write_bytes(audio_data)
    tts_ms = (time.perf_counter() - tts_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "transcript": text,
        "response": response_text,
        "audio_url": f"/audio/{cache_path.name}",
        "bridge_ms": bridge_ms,
        "tts_ms": tts_ms,
        "total_ms": total_ms,
    }


# ── Audio serving ─────────────────────────────────────────────────────

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = CACHE_DIR / filename
    if not filepath.exists():
        return {"error": "Audio not found"}
    return FileResponse(filepath, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)
