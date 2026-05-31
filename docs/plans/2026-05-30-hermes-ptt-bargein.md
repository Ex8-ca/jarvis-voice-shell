# Hermes CLI Brain, Real PTT, and Barge-In Implementation Plan

> **For Hermes:** Implement directly with TDD and verify with real commands on Windows.

**Goal:** Turn the current typed JARVIS wrapper into a usable voice shell that can talk to Hermes Agent, capture real push-to-talk microphone turns, and stop speech when interrupted.

**Architecture:** Keep the realtime shell separate from the reasoning core. Add a Hermes CLI subprocess bridge for “Hermes aka me,” keep the existing HTTP bridge for raw local Qwen, and route both through one turn processor. Add a keyboard-backed PTT controller with a portable typed fallback. Add cancellation-aware TTS playback and a barge-in path that cancels current speech before accepting a new turn.

**Tech Stack:** Python 3.11+, click, asyncio, keyboard optional dependency, sounddevice/PyAudio recording fallback, Whisper STT, Edge TTS, Hermes CLI.

---

### Task 1: Add Hermes CLI bridge

**Objective:** Let the wrapper call real Hermes Agent via `hermes chat -Q -q <transcript>` instead of only raw OpenAI-compatible HTTP.

**Files:**
- Modify: `src/jarvis_voice_shell/config.py`
- Modify: `src/jarvis_voice_shell/bridge.py`
- Test: `tests/test_bridge.py`

**Steps:**
1. Write tests for command construction, successful stdout, non-zero exit raising `BridgeError`, and token callback invocation.
2. Verify tests fail because `HermesCliBridge` does not exist.
3. Implement `HermesCliBridge` with injectable subprocess runner and `send()` API matching existing bridges.
4. Run bridge tests and full suite.

### Task 2: Add PTT controller abstraction

**Objective:** Provide a testable keyboard hotkey abstraction with graceful fallback if the `keyboard` package is missing or lacks privileges.

**Files:**
- Create: `src/jarvis_voice_shell/ptt.py`
- Test: `tests/test_ptt.py`

**Steps:**
1. Write tests for event classification: press starts listening, release stops listening, interrupt key cancels playback, unavailable backend reports typed fallback.
2. Verify tests fail because `ptt.py` does not exist.
3. Implement `PTTEvent`, `KeyboardPTTController`, and `TypedPTTController`.
4. Run PTT tests and full suite.

### Task 3: Add bounded PTT recording support

**Objective:** Record while the PTT key is held, with a max duration safety cap and existing fixed-duration recording preserved.

**Files:**
- Modify: `src/jarvis_voice_shell/recorder.py`
- Test: `tests/test_recorder.py`

**Steps:**
1. Write tests around stop-event-driven frame collection using fake PyAudio stream.
2. Verify tests fail because `record_until()` does not exist.
3. Implement `record_until(stop_event, output_path, max_seconds, input_device_index)` for PyAudio and a safe sounddevice fallback that records short bounded windows.
4. Run recorder tests and full suite.

### Task 4: Extract turn processing

**Objective:** Reuse the same record/transcribe/brain/TTS pipeline from typed mode, listen-once, and PTT mode.

**Files:**
- Modify: `src/jarvis_voice_shell/cli.py`
- Test: `tests/test_cli_turns.py`

**Steps:**
1. Write tests for bridge selection (`echo`, `http`, `hermes-cli`) and one turn processor behavior using fake bridge/TTS/STT.
2. Verify tests fail.
3. Implement helper functions to select bridge and process a transcript.
4. Run CLI-turn tests and full suite.

### Task 5: Implement real PTT run mode

**Objective:** Make `jarvis_voice_shell.cli run` use the keyboard hotkey when available: hold key, record mic, release, transcribe, send to selected brain, speak.

**Files:**
- Modify: `src/jarvis_voice_shell/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli_ptt.py`

**Steps:**
1. Write tests for `--input-mode typed|ptt`, `--brain http|hermes-cli|echo`, and fallback messages.
2. Verify tests fail.
3. Add CLI options and route to typed loop or PTT loop.
4. Add `keyboard` optional dependency.
5. Run CLI tests and full suite.

### Task 6: Implement barge-in cancellation

**Objective:** When the user presses PTT while JARVIS is speaking, stop playback immediately and supersede the current turn.

**Files:**
- Modify: `src/jarvis_voice_shell/tts.py`
- Modify: `src/jarvis_voice_shell/cli.py`
- Test: `tests/test_tts.py`, `tests/test_cli_ptt.py`

**Steps:**
1. Write tests proving `cancel_playback()` stops sounddevice playback via `sd.stop()` and PyAudio playback checks a cancellation flag between chunks.
2. Verify tests fail for sounddevice stop and chunk cancellation behavior if missing.
3. Implement a playback cancellation event and backend stop hooks.
4. Wire PTT press during speaking to `tts.cancel_playback()`.
5. Run TTS/PTT tests and full suite.

### Task 7: Live verification

**Objective:** Prove the project works on this Windows box.

**Commands:**
- `python -m pytest -q` expected: all tests pass.
- `python -m jarvis_voice_shell.cli test-bridge --url http://127.0.0.1:8642/v1/chat/completions --text "confirm online"` expected: Hermes Gateway response.
- `python -m jarvis_voice_shell.cli run --input-mode typed --brain http` expected: typed input goes to Hermes Gateway and speaks reply.
- `python -m jarvis_voice_shell.cli listen-once --seconds 5 --stt-engine whisper --stt-model tiny --brain http` expected: real mic turn path works if audio auto-detection selects the right devices.
- `python -m jarvis_voice_shell.cli run --input-mode ptt --brain http --stt-engine whisper --stt-model tiny` expected: hold PTT to speak, release to send, press PTT during speech to stop it. If auto-detection picks the wrong route, pass local `--input-device` and `--output-device` values without committing them.
