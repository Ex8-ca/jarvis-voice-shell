"""
JARVIS Voice Web Interface
Simple Flask app that provides a web UI for testing JARVIS Voice Shell from any device on the LAN.
"""

import asyncio
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-web")

app = FastAPI(title="JARVIS Voice Web")

# Hermes gateway config
HERMES_URL = "http://192.168.1.3:6789/v1/chat/completions"
HERMES_API_KEY = "chillygeek6789"
HERMES_MODEL = "hermes-agent"

# TTS cache directory
CACHE_DIR = Path.home() / ".cache" / "jarvis-voice-shell" / "tts_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content="""
<!DOCTYPE html>
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
        .container {
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 12px;
            background: #1a1a2e;
            border: 1px solid #333;
        }
        .status.listening { border-color: #00ff88; box-shadow: 0 0 20px rgba(0,255,136,0.3); }
        .status.thinking { border-color: #ffd700; box-shadow: 0 0 20px rgba(255,215,0,0.3); }
        .status.speaking { border-color: #00d4ff; box-shadow: 0 0 20px rgba(0,212,255,0.3); }
        .btn {
            display: inline-block;
            padding: 20px 40px;
            margin: 20px 0;
            font-size: 1.5rem;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 120px;
            height: 120px;
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: #000;
            font-weight: bold;
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
            margin-top: 20px;
            text-align: left;
            max-height: 60vh;
            overflow-y: auto;
        }
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
        }
        .message.user {
            background: #1a1a2e;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .message.jarvis {
            background: #2a2a3e;
            border-bottom-left-radius: 4px;
        }
        .message .label {
            font-size: 0.8rem;
            color: #888;
            margin-bottom: 4px;
        }
        .typing { font-style: italic; color: #888; }
        .latency {
            font-size: 0.75rem;
            color: #666;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>JARVIS</h1>
        <p style="color: #666; margin-bottom: 20px;">Voice Assistant — Tap to Speak</p>
        
        <div class="status" id="status">Ready</div>
        
        <button class="btn" id="mic-btn" onclick="toggleVoice()">🎤</button>
        
        <div style="margin: 15px 0;">
            <input type="text" id="text-input" placeholder="Type your message..." 
                   style="width: 70%; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a2e; color: #e0e0e0; font-size: 1rem;">
            <button onclick="sendText()" 
                    style="padding: 12px 20px; border-radius: 8px; border: 1px solid #333; background: #00d4ff; color: #000; font-weight: bold; cursor: pointer;">Send</button>
        </div>
        
        <p style="font-size: 0.8rem; color: #666;">💡 Mic requires HTTPS. Use text input for now.</p>
        
        <div class="conversation" id="conversation"></div>
        
        <button onclick="clearChat()" style="margin-top: 20px; padding: 8px 16px; background: #333; color: #888; border: 1px solid #444; border-radius: 8px; cursor: pointer;">Clear Chat</button>
    </div>
    
    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isListening = false;
        let audioContext = null;
        
        // Load conversation history from localStorage
        let conversation = JSON.parse(localStorage.getItem('jarvis_convo') || '[]');
        renderConversation();
        
        async function toggleVoice() {
            const btn = document.getElementById('mic-btn');
            const status = document.getElementById('status');
            
            if (isListening) {
                // Stop listening
                mediaRecorder.stop();
                isListening = false;
                btn.classList.remove('listening');
                btn.innerHTML = '🎤';
                status.className = 'status';
                status.textContent = 'Processing...';
            } else {
                // Start listening
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioContext = new AudioContext();
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        stream.getTracks().forEach(t => t.stop());
                        await processAudio();
                    };
                    
                    mediaRecorder.start();
                    isListening = true;
                    btn.classList.add('listening');
                    btn.innerHTML = '🔴';
                    status.className = 'status listening';
                    status.textContent = 'Listening...';
                } catch (err) {
                    status.textContent = 'Mic error: ' + err.message;
                    console.error(err);
                }
            }
        }
        
        async function processText(text) {
            const status = document.getElementById('status');
            status.className = 'status thinking';
            status.textContent = 'Thinking...';
            
            addMessage(text, 'user');
            
            const start = performance.now();
            
            try {
                const response = await fetch(`/chat?text=${encodeURIComponent(text)}`);
                const data = await response.json();
                const elapsed = ((performance.now() - start) / 1000).toFixed(2);
                
                addMessage(data.response, 'jarvis', {
                    bridge_ms: data.bridge_ms,
                    tts_ms: data.tts_ms,
                    total_ms: data.total_ms
                });
                
                // Play TTS audio
                if (data.audio_url) {
                    status.className = 'status speaking';
                    status.textContent = 'Speaking...';
                    
                    const audio = new Audio(data.audio_url);
                    audio.play();
                    
                    audio.onended = () => {
                        status.className = 'status';
                        status.textContent = 'Ready';
                    };
                }
            } catch (err) {
                status.textContent = 'Error: ' + err.message;
                console.error(err);
            }
        }
        
        async function processAudio() {
            const status = document.getElementById('status');
            status.className = 'status thinking';
            status.textContent = 'Processing...';
            
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            const start = performance.now();
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                const elapsed = ((performance.now() - start) / 1000).toFixed(2);
                
                addMessage(data.transcript || '(no speech)', 'user');
                addMessage(data.response, 'jarvis', {
                    bridge_ms: data.bridge_ms,
                    tts_ms: data.tts_ms,
                    total_ms: data.total_ms
                });
                
                // Play TTS audio
                if (data.audio_url) {
                    status.className = 'status speaking';
                    status.textContent = 'Speaking...';
                    
                    const audio = new Audio(data.audio_url);
                    await audio.play();
                    
                    status.className = 'status';
                    status.textContent = 'Ready';
                }
            } catch (err) {
                status.textContent = 'Error: ' + err.message;
                console.error(err);
            }
        }
        
        function addMessage(text, role, latency = null) {
            conversation.push({ text, role, latency, time: Date.now() });
            localStorage.setItem('jarvis_convo', JSON.stringify(conversation));
            renderConversation();
        }
        
        function renderConversation() {
            const container = document.getElementById('conversation');
            container.innerHTML = conversation.map(msg => `
                <div class="message ${msg.role}">
                    <div class="label">${msg.role === 'user' ? 'You' : 'JARVIS'}</div>
                    <div>${msg.text}</div>
                    ${msg.latency ? `<div class="latency">⏱ ${msg.latency.total_ms?.toFixed(0) || '?'}ms total</div>` : ''}
                </div>
            `).join('');
            container.scrollTop = container.scrollHeight;
        }
        
        function clearChat() {
            conversation = [];
            localStorage.removeItem('jarvis_convo');
            document.getElementById('conversation').innerHTML = '';
        }
        
        async function sendText() {
            const input = document.getElementById('text-input');
            const text = input.value.trim();
            if (!text) return;
            
            input.value = '';
            await processText(text);
        }
        
        // Allow Enter key to send
        document.getElementById('text-input')?.addEventListener('keypress', e => {
            if (e.key === 'Enter') sendText();
        });
    </script>
</body>
</html>
""")


@app.post("/process")
async def process_audio():
    """Process audio: STT → Hermes LLM → TTS → return response + audio."""
    import time
    import hashlib
    
    total_start = time.perf_counter()
    
    # Note: For now, this endpoint expects audio file uploads but we'll use
    # the browser's built-in Speech Recognition for STT to keep it simple.
    # The actual flow will be handled by JavaScript sending text directly.
    
    return {
        "error": "Use /chat endpoint with text input",
        "status": "not_implemented"
    }


@app.post("/chat")
async def chat(text: str):
    """Process text input: Hermes LLM → TTS → return response + audio URL."""
    import time
    
    total_start = time.perf_counter()
    
    # Step 1: Call Hermes LLM
    bridge_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            HERMES_URL,
            json={
                "model": HERMES_MODEL,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 300,
                "stream": False,
            },
            headers={
                "Authorization": f"Bearer {HERMES_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
    
    bridge_ms = (time.perf_counter() - bridge_start) * 1000
    
    # Step 2: Generate TTS using edge-tts
    tts_start = time.perf_counter()
    import edge_tts
    import hashlib
    
    text_hash = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"tts_{text_hash}.mp3"
    
    if cache_path.exists():
        tts_ms = (time.perf_counter() - tts_start) * 1000
    else:
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


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve cached TTS audio files."""
    from fastapi.responses import FileResponse
    
    filepath = CACHE_DIR / filename
    if not filepath.exists():
        return {"error": "Audio not found"}
    
    return FileResponse(filepath, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)
