"""CLI entry point for JARVIS Voice Shell.

Commands:
    jarvis-voice run          Start the push-to-talk voice loop.
    jarvis-voice list-devices   Enumerate audio input/output devices.
    jarvis-voice test-tts     Synthesize a test phrase to verify TTS chain.
    jarvis-voice test-bridge  Send a test transcript to Hermes (no audio).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import click

from .audio_devices import AudioDeviceManager
from .bridge import EchoBridge, HermesBridge, HermesCliBridge
from .config import Config
from .controller import ConversationController, ConversationState, TurnContext
from .latency import LatencyLogger, LatencyTracker, TurnLatency
from .openai_voice import MissingOpenAIAPIKey, OpenAIRealtimeVoiceClient, OpenAIRealtimeVoiceConfig
from .recorder import AudioRecorder, RecorderError
from .speech_queue import InterruptibleSpeechQueue
from .stt import STTEngine
from .tts import TTSEngine
from .vad import EnergyVAD
from .voice_profile import is_status_query, sanitize_for_speech, spoken_failure_message, wrap_for_voice

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="jarvis-voice")
def main():
    """JARVIS Voice Shell — realtime push-to-talk voice interface for Hermes."""


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--input-device", "-i", type=int, default=None,
    help="PyAudio input device index (auto-detect if not set).",
)
@click.option(
    "--output-device", "-o", type=int, default=None,
    help="PyAudio output device index (auto-detect if not set).",
)
@click.option(
    "--tts-voice", type=str, default=None,
    help="Edge TTS voice name (default: en-GB-RyanNeural).",
)
@click.option(
    "--tts-rate", type=str, default=None,
    help="Edge TTS speech rate modifier, e.g. '+10%' for slightly faster speech.",
)
@click.option(
    "--bridge-url", type=str, default=None,
    help="Hermes Gateway API URL (default: http://127.0.0.1:8642/v1/chat/completions).",
)
@click.option(
    "--ptt-key", type=str, default=None,
    help="Push-to-talk hotkey (default: `).",
)
@click.option(
    "--input-mode",
    type=click.Choice(["typed", "ptt", "always-on"]),
    default="typed",
    help="Input mode: typed fallback, push-to-talk, or always-on VAD full duplex.",
)
@click.option(
    "--brain",
    type=click.Choice(["http", "hermes-cli", "echo"]),
    default="http",
    help="Reasoning backend: Hermes Gateway HTTP, slower Hermes CLI, or echo test.",
)
@click.option("--sample-rate", type=int, default=None, help="Microphone sample rate override, e.g. 48000 for some headset mics.")
@click.option("--stt-engine", type=click.Choice(["stub", "whisper"]), default="whisper")
@click.option("--stt-model", type=str, default="tiny")
@click.option("--max-record-seconds", type=float, default=30.0)
@click.option("--vad-threshold", type=int, default=None, help="Always-on VAD RMS threshold (default: 60).")
@click.option("--vad-end-silence-ms", type=int, default=None, help="Always-on VAD trailing silence before send.")
def run(
    verbose: bool,
    input_device: int | None,
    output_device: int | None,
    tts_voice: str | None,
    tts_rate: str | None,
    bridge_url: str | None,
    ptt_key: str | None,
    input_mode: str,
    brain: str,
    sample_rate: int | None,
    stt_engine: str,
    stt_model: str,
    max_record_seconds: float,
    vad_threshold: int | None,
    vad_end_silence_ms: int | None,
):
    """Start the JARVIS push-to-talk voice loop.

    Press the PTT hotkey, speak, release — JARVIS responds via TTS.
    Press Ctrl+C or the configured barge-in key to interrupt playback.
    """
    _setup_logging(verbose)

    # Build config: defaults → env → CLI overrides
    config = Config.from_env()
    overrides = {}
    if input_device is not None:
        overrides["input_device_index"] = input_device
    if output_device is not None:
        overrides["tts_output_device_index"] = output_device
    if tts_voice is not None:
        overrides["tts_voice"] = tts_voice
    if tts_rate is not None:
        overrides["tts_rate"] = tts_rate
    if bridge_url is not None:
        overrides["hermes_bridge_url"] = bridge_url
    if ptt_key is not None:
        overrides["ptt_key"] = ptt_key
    if sample_rate is not None:
        overrides["sample_rate"] = sample_rate
    overrides["stt_engine"] = stt_engine
    overrides["stt_model"] = stt_model
    if vad_threshold is not None:
        overrides["vad_energy_threshold"] = vad_threshold
    if vad_end_silence_ms is not None:
        overrides["vad_end_silence_ms"] = vad_end_silence_ms
    config = config.replace(**overrides)

    click.echo("╔══════════════════════════════════════════╗")
    click.echo("║     JARVIS Voice Shell v0.1.0           ║")
    click.echo("╠══════════════════════════════════════════╣")
    click.echo(f"║  PTT key:     {config.ptt_key:<28s}║")
    click.echo(f"║  TTS engine:  {config.tts_engine:<28s}║")
    click.echo(f"║  TTS voice:   {config.tts_voice:<28s}║")
    click.echo(f"║  TTS rate:    {config.tts_rate:<28s}║")
    click.echo(f"║  Bridge:      {config.hermes_bridge_url:<28s}║")
    click.echo("╚══════════════════════════════════════════╝")

    # Enumerate devices
    dev_mgr = AudioDeviceManager()
    dev_mgr.refresh()

    try:
        selected_input = dev_mgr.select_input(config.input_device_index)
        click.echo(f"\nInput:  {selected_input.display_name}")
    except RuntimeError as e:
        click.echo(f"\nInput:  ERROR — {e}", err=True)
        click.echo("Connect a microphone and retry.", err=True)
        sys.exit(1)

    try:
        selected_output = dev_mgr.select_output(config.tts_output_device_index)
        click.echo(f"Output: {selected_output.display_name}")
    except RuntimeError as e:
        click.echo(f"Output: ERROR — {e}", err=True)
        click.echo("Connect speakers/headset and retry.", err=True)
        sys.exit(1)

    click.echo(f"\nBridge: {config.hermes_bridge_url}")
    click.echo(f"Brain:  {brain}")
    click.echo(f"Mode:   {input_mode}")
    click.echo("\nReady. Press PTT key to speak, Ctrl+C to exit.\n")

    asyncio.run(_voice_loop(config, dev_mgr, selected_input, selected_output, input_mode, brain, max_record_seconds))


def _make_bridge(config: Config, brain: str):
    """Create the selected reasoning backend."""
    if brain == "echo":
        return EchoBridge(chunk_size=12)
    if brain == "hermes-cli":
        return HermesCliBridge()
    return HermesBridge(config)


async def _voice_loop(
    config: Config,
    dev_mgr: AudioDeviceManager,
    input_device,
    output_device,
    input_mode: str,
    brain: str,
    max_record_seconds: float,
) -> None:
    """Main voice loop: typed fallback or real push-to-talk microphone capture."""
    bridge = _make_bridge(config, brain)
    tts = TTSEngine(config)
    stt = STTEngine(config)
    recorder = AudioRecorder(config)
    tracker = LatencyTracker()

    active_tts_latency: TurnLatency | None = None

    async def _speak_queue_chunk(text: str) -> None:
        await tts.speak(text, latency=active_tts_latency)

    speech_queue = InterruptibleSpeechQueue(speak=_speak_queue_chunk)

    async def _cancel_audio() -> None:
        # Kill the audio output FIRST (sd.stop unblocks sd.wait in the playback
        # thread), then drain the speech queue. Reversing this order caused a
        # deadlock: speech_queue.interrupt() awaited the stuck task, but the
        # task was blocked in sd.wait() which only sd.stop() can unblock.
        await tts.cancel_playback()
        await speech_queue.interrupt()

    controller = ConversationController(cancel_playback=_cancel_audio)
    await speech_queue.start()

    def on_token(full_text: str, delta: str) -> None:
        # In v1 we accumulate and TTS the full response. Streaming TTS is a v2 goal.
        pass

    async def _process_turn(transcript: str, turn: TurnLatency | None = None, ctx: TurnContext | None = None) -> None:
        nonlocal active_tts_latency
        ctx = ctx or controller.start_turn()
        turn = turn or tracker.new_turn()
        if not controller.is_current(ctx.turn_id):
            tracker.discard_turn(turn)
            return
        if turn.stt_start == 0.0:
            turn.stt_start = turn.turn_start
        if turn.stt_end == 0.0:
            turn.stt_end = turn.turn_start
        controller.set_state(ConversationState.THINKING, ctx.turn_id)
        click.echo(f"\n  You: {transcript}")
        try:
            if is_status_query(transcript):
                await bridge.send(wrap_for_voice("Health check. Reply with exactly: Hermes online."), latency=turn)
                response = "Online, sir. Voice loop active and Hermes bridge responding."
            else:
                response = await bridge.send(wrap_for_voice(transcript), on_token=on_token, latency=turn)
            if not controller.is_current(ctx.turn_id):
                tracker.discard_turn(turn)
                return
            speech_response = sanitize_for_speech(response)
            click.echo(f"  JARVIS: {response}")
            if speech_response != response:
                click.echo(f"  Spoken: {speech_response}")
            controller.set_state(ConversationState.SPEAKING, ctx.turn_id)
            active_tts_latency = turn
            await speech_queue.enqueue(speech_response or "Done, sir.")
            await speech_queue.join()
        except asyncio.CancelledError:
            tracker.discard_turn(turn)
            raise
        except Exception as e:
            logger.error("Turn failed: %s", e)
            click.echo(f"  ERROR: {e}", err=True)
            try:
                await speech_queue.enqueue(spoken_failure_message(e))
                await speech_queue.join()
            except Exception:
                logger.exception("Failed to speak failure alert")
        finally:
            if active_tts_latency is turn:
                active_tts_latency = None
            if controller.is_current(ctx.turn_id):
                controller.set_state(ConversationState.IDLE, ctx.turn_id)
                tracker.finish_turn(turn)
                click.echo(f"  [TtFA: {turn.total_ttfa_ms:.0f}ms]\n")

    async def _typed_loop() -> None:
        click.echo("Typed fallback active. Type a message and press Enter (Ctrl+C to quit).\n")
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            text = line.strip()
            if text:
                await _process_turn(text)

    async def _ptt_loop() -> None:
        try:
            import keyboard  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Real PTT requires the keyboard package: python -m pip install keyboard") from exc

        click.echo(f"Real PTT active. Hold {config.ptt_key}, speak, release to send.")
        click.echo("Press the PTT key while JARVIS is speaking to barge in / stop playback. Ctrl+C exits.\n")
        was_pressed = False
        stop_event: threading.Event | None = None
        record_task: asyncio.Task | None = None
        record_path = config.cache_dir / "recordings" / "ptt.wav"
        active_ctx: TurnContext | None = None
        active_turn: TurnLatency | None = None
        while True:
            pressed = keyboard.is_pressed(config.ptt_key)
            if pressed and not was_pressed:
                await controller.interrupt()
                active_ctx = controller.start_turn()
                active_turn = tracker.new_turn()
                stop_event = threading.Event()
                click.echo("Listening...")
                record_task = asyncio.create_task(asyncio.to_thread(
                    recorder.record_until,
                    stop_event,
                    record_path,
                    input_device.index,
                    max_record_seconds,
                    active_turn,
                ))
                controller.track_task(record_task, active_ctx.turn_id)
            if was_pressed and not pressed and stop_event is not None and record_task is not None:
                stop_event.set()
                try:
                    result = await record_task
                except RecorderError as exc:
                    if active_turn is not None:
                        tracker.discard_turn(active_turn)
                    click.echo(f"Recording failed: {exc}", err=True)
                    click.echo("Tip: run `python -m jarvis_voice_shell.cli list-devices` and try another input device index.", err=True)
                    stop_event = None
                    record_task = None
                    active_ctx = None
                    active_turn = None
                    was_pressed = pressed
                    await asyncio.sleep(0.03)
                    continue
                click.echo(f"Recorded: {result.path}")
                if active_ctx is not None:
                    controller.set_state(ConversationState.TRANSCRIBING, active_ctx.turn_id)
                transcript = stt.transcribe_file(result.path).text
                if transcript and active_ctx is not None and active_turn is not None:
                    task = asyncio.create_task(_process_turn(transcript, active_turn, active_ctx))
                    controller.track_task(task, active_ctx.turn_id)
                else:
                    if active_turn is not None:
                        tracker.discard_turn(active_turn)
                    click.echo("Transcript was empty; nothing sent.")
                stop_event = None
                record_task = None
                active_ctx = None
                active_turn = None
            was_pressed = pressed
            await asyncio.sleep(0.03)

    async def _always_on_loop() -> None:
        click.echo("Always-on full duplex active. Speak naturally; JARVIS will stop when you start talking.")
        click.echo(f"VAD threshold: {config.vad_energy_threshold}; end silence: {config.vad_end_silence_ms}ms. Ctrl+C exits.")
        click.echo("If you see 'No speech detected', compare peak RMS to threshold: speaking must peak above threshold.\n")
        stop_event = threading.Event()
        interrupt_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        record_path = config.cache_dir / "recordings" / "always-on.wav"

        async def _interrupt_watcher() -> None:
            while not stop_event.is_set():
                await interrupt_event.wait()
                interrupt_event.clear()
                await controller.interrupt()

        watcher = asyncio.create_task(_interrupt_watcher())

        async def _startup_health_check() -> None:
            try:
                await bridge.send(wrap_for_voice("Startup health check. Reply with exactly: Hermes online."))
                await speech_queue.enqueue("Hermes online.")
            except Exception as exc:
                click.echo(f"Startup health check failed: {exc}", err=True)
                try:
                    await speech_queue.enqueue(spoken_failure_message(exc))
                except Exception:
                    logger.exception("Failed to speak startup health alert")

        startup_health_task: asyncio.Task | None = None
        # Always speak the armed confirmation and run the health check,
        # regardless of brain type. HTTP brain was previously skipped,
        # leaving the user in silence on startup.
        await speech_queue.enqueue("Voice loop armed, sir.")
        await speech_queue.join()
        startup_health_task = asyncio.create_task(_startup_health_check())
        _empty_transcript_streak = 0
        try:
            while True:
                # Do not start a new controller turn until a complete utterance has
                # been captured. While listening, the previous response must remain
                # the current turn so speech-start can interrupt active playback.
                turn = tracker.new_turn()
                vad = EnergyVAD(
                    energy_threshold=config.vad_energy_threshold,
                    start_frames=config.vad_start_frames,
                    end_silence_frames=max(1, int((config.vad_end_silence_ms / 1000) * config.sample_rate / config.chunk_size)),
                    pre_roll_frames=max(0, int((config.vad_pre_roll_ms / 1000) * config.sample_rate / config.chunk_size)),
                )
                listen_start = time.perf_counter()
                try:
                    result = await asyncio.to_thread(
                        recorder.record_vad_segment,
                        stop_event,
                        record_path,
                        input_device.index,
                        max_record_seconds,
                        turn,
                        vad,
                        lambda: loop.call_soon_threadsafe(interrupt_event.set),
                    )
                except RecorderError as exc:
                    tracker.discard_turn(turn)
                    listen_elapsed = time.perf_counter() - listen_start
                    if "No speech detected" in str(exc):
                        click.echo(f"Listening... ({exc})")
                        if listen_elapsed < 1.0:
                            click.echo(
                                "Input returned silence immediately; this device route is probably inactive. "
                                "Try another input device index from list-devices.",
                                err=True,
                            )
                            await asyncio.sleep(1.0)
                        else:
                            await asyncio.sleep(0.25)
                    else:
                        click.echo(f"VAD recording failed: {exc}", err=True)
                        click.echo("Audio device open failed; backing off 2s. If this repeats, close other JARVIS windows or use list-devices.", err=True)
                        await asyncio.sleep(2.0)
                    continue
                click.echo(f"Heard: {result.path}")
                transcript = stt.transcribe_file(result.path).text
                if transcript:
                    _empty_transcript_streak = 0
                    ctx = controller.start_turn()
                    controller.set_state(ConversationState.TRANSCRIBING, ctx.turn_id)
                    task = asyncio.create_task(_process_turn(transcript, turn, ctx))
                    controller.track_task(task, ctx.turn_id)
                else:
                    tracker.discard_turn(turn)
                    _empty_transcript_streak += 1
                    # Back off on repeated empty transcripts — prevents CPU spinning
                    # when VAD triggers on noise but Whisper finds no speech.
                    if _empty_transcript_streak >= 3:
                        backoff = min(_empty_transcript_streak * 0.5, 5.0)
                        click.echo(f"Empty transcript x{_empty_transcript_streak}; backing off {backoff:.1f}s.")
                        await asyncio.sleep(backoff)
                    else:
                        click.echo("Transcript was empty; listening continues.")
        finally:
            stop_event.set()
            watcher.cancel()
            pending_tasks = [watcher]
            if startup_health_task is not None:
                startup_health_task.cancel()
                pending_tasks.append(startup_health_task)
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    try:
        if input_mode == "ptt":
            await _ptt_loop()
        elif input_mode == "always-on":
            await _always_on_loop()
        else:
            await _typed_loop()
    except asyncio.CancelledError:
        pass
    finally:
        click.echo(f"\n{tracker.summary()}")
        recorder.close()
        await speech_queue.close()
        await tts.close()
        await bridge.close()


@main.command("openai-voice")
@click.option("--input-device", "-i", type=int, default=None, help="Input device index.")
@click.option("--output-device", "-o", type=int, default=None, help="Output device index.")
@click.option("--sample-rate", type=int, default=24000, help="Device sample rate; 24000 avoids resampling OpenAI PCM.")
@click.option("--model", type=str, default="gpt-realtime-mini", help="Small OpenAI Realtime voice-layer model.")
@click.option("--voice", type=str, default="ash", help="OpenAI Realtime voice.")
@click.option("--instructions", type=str, default="You are only JARVIS' realtime ears and mouth. Do not answer independently. Wait for Hermes text, then speak it exactly.")
@click.option("--brain", type=click.Choice(["hermes-cli", "echo"]), default="hermes-cli", help="Brain behind the voice layer.")
@click.option("--dry-run", is_flag=True, help="Validate configuration without opening audio/websocket.")
def openai_voice(
    input_device: int | None,
    output_device: int | None,
    sample_rate: int,
    model: str,
    voice: str,
    instructions: str,
    brain: str,
    dry_run: bool,
):
    """Start OpenAI Realtime voice mode.

    OpenAI Realtime owns VAD, full-duplex turn-taking, interruption, STT, and TTS.
    """
    try:
        cfg = OpenAIRealtimeVoiceConfig.from_env(
            model=model,
            voice=voice,
            instructions=instructions,
            input_device_index=input_device,
            output_device_index=output_device,
            device_sample_rate=sample_rate,
        )
    except MissingOpenAIAPIKey as exc:
        click.echo(str(exc), err=True)
        raise click.Abort() from exc

    click.echo("OpenAI Realtime voice mode ready.")
    click.echo(f"Model:  {cfg.model}")
    click.echo(f"Voice:  {cfg.voice}")
    click.echo(f"Input:  {cfg.input_device_index if cfg.input_device_index is not None else 'system default'}")
    click.echo(f"Output: {cfg.output_device_index if cfg.output_device_index is not None else 'system default'}")
    click.echo(f"Sample: {cfg.device_sample_rate} Hz")
    click.echo(f"Brain:  {brain} (Hermes owns reasoning/tools/memory)" if brain == "hermes-cli" else "Brain:  echo test mode")
    if dry_run:
        return
    bridge = HermesCliBridge() if brain == "hermes-cli" else EchoBridge(mode="echo")
    click.echo("Connecting. OpenAI is ears/mouth only; Hermes is the brain. Ctrl+C exits.")
    try:
        asyncio.run(OpenAIRealtimeVoiceClient(cfg, on_status=click.echo, brain_bridge=bridge).run())
    except KeyboardInterrupt:
        click.echo("Stopped.")


@main.command("local-voice")
@click.option("--input-device", "-i", type=int, default=None, help="Input device index.")
@click.option("--output-device", "-o", type=int, default=None, help="Output device index.")
@click.option("--sample-rate", type=int, default=16000, help="Microphone sample rate (16000 for whisper).")
@click.option("--voice", type=str, default=None, help="Edge TTS voice name (default: en-GB-RyanNeural).")
@click.option("--brain", type=click.Choice(["hermes-cli", "echo"]), default="hermes-cli", help="Brain behind the voice layer.")
@click.option("--stt-model", type=str, default="tiny", help="Whisper model name.")
@click.option("--vad-threshold", type=int, default=60, help="VAD RMS threshold.")
@click.option("--vad-end-silence-ms", type=int, default=700, help="Trailing silence before stop.")
@click.option("--dry-run", is_flag=True, help="Validate without opening audio.")
@click.option("--jarvis", "voice_style", flag_value="jarvis", help="Use Jarvis-optimized voice (en-GB-RyanNeural).")
@click.option("--friday", "voice_style", flag_value="friday", help="Use Friday-optimized voice (en-US-GuyNeural).")
def local_voice(
    input_device,
    output_device,
    sample_rate,
    voice,
    brain,
    stt_model,
    vad_threshold,
    vad_end_silence_ms,
    dry_run,
    voice_style,
):
    """Start fully local voice mode (no OpenAI API).

    Pipeline: Energy VAD → Whisper (local) → Hermes CLI → Edge TTS (local).
    No API keys required. Runs entirely on your machine.
    """
    _setup_logging(True)
    from .local_voice import LocalVoiceClient, _default_voice_for_jarvis

    config = Config.from_env()
    overrides = {}
    if input_device is not None:
        overrides["input_device_index"] = input_device
    if output_device is not None:
        overrides["tts_output_device_index"] = output_device
    overrides["sample_rate"] = sample_rate
    overrides["stt_engine"] = "whisper"
    overrides["stt_model"] = stt_model
    overrides["vad_energy_threshold"] = vad_threshold
    overrides["vad_end_silence_ms"] = vad_end_silence_ms

    if voice_style == "jarvis":
        overrides["tts_voice"] = _default_voice_for_jarvis()
    elif voice_style == "friday":
        overrides["tts_voice"] = "en-US-GuyNeural"
    elif voice is not None:
        overrides["tts_voice"] = voice

    config = config.replace(**overrides)

    click.echo("╔══════════════════════════════════════════╗")
    click.echo("║    JARVIS Local Voice — fully offline    ║")
    click.echo("╠══════════════════════════════════════════╣")
    click.echo(f"║  STT:       whisper ({stt_model:<6s})         ║")
    click.echo(f"║  TTS:       edge-tts ({config.tts_voice:<19s})║")
    click.echo(f"║  Brain:     {brain:<28s}║")
    click.echo(f"║  VAD:       threshold {vad_threshold}         ║")
    click.echo("╚══════════════════════════════════════════╝")

    if dry_run:
        click.echo("Dry run — everything looks good.")
        return

    if brain == "hermes-cli":
        from .bridge import HermesCliBridge
        bridge = HermesCliBridge()
    else:
        from .bridge import EchoBridge
        bridge = EchoBridge(mode="echo")

    client = LocalVoiceClient(config, on_status=click.echo, bridge=bridge)
    click.echo("Listening... (energy VAD, Ctrl+C exits)\n")
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        client.stop()
        click.echo("Stopped.")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (machine-readable).")
def list_devices(as_json: bool):
    """List all available audio input and output devices with classification."""
    _setup_logging(verbose=False)
    dev_mgr = AudioDeviceManager()
    dev_mgr.refresh()

    if as_json:
        import json
        devices = dev_mgr.list_devices_json()
        click.echo(json.dumps(devices, indent=2))
    else:
        for line in dev_mgr.list_devices():
            click.echo(line)


@main.command()
@click.option("--voice", type=str, default="en-GB-RyanNeural",
              help="Edge TTS voice name.")
@click.option("--text", type=str, default="Hello, sir. JARVIS voice shell is operational.",
              help="Text to synthesize.")
def test_tts(voice: str, text: str):
    """Test TTS by speaking a phrase."""
    _setup_logging(verbose=True)
    config = Config.from_env().replace(tts_voice=voice)
    tts = TTSEngine(config)
    click.echo(f"Synthesizing: '{text}' with voice {voice}...")
    asyncio.run(tts.speak(text))
    asyncio.run(tts.close())
    click.echo("Done.")


@main.command()
@click.option("--input-device", "-i", type=int, default=None,
              help="PyAudio input device index. Auto-selects headset if omitted.")
@click.option("--seconds", "-s", type=float, default=3.0,
              help="Seconds to record.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output WAV path. Defaults to cache_dir/recordings/probe.wav.")
def record_test(input_device: int | None, seconds: float, output: Path | None):
    """Record a short WAV clip from the selected microphone."""
    _setup_logging(verbose=True)
    config = Config.from_env()
    dev_mgr = AudioDeviceManager()
    dev_mgr.refresh()
    selected = dev_mgr.select_input(input_device)
    output_path = output or (config.cache_dir / "recordings" / "probe.wav")
    click.echo(f"Recording {seconds:.1f}s from {selected.display_name} -> {output_path}")
    recorder = AudioRecorder(config)
    try:
        result = recorder.record_seconds(seconds, output_path, selected.index)
    finally:
        recorder.close()
    click.echo(
        f"Wrote {result.path} "
        f"({result.frames} frames, {result.sample_rate} Hz, {result.channels} ch)."
    )


@main.command()
@click.option("--input-device", "-i", type=int, default=None,
              help="Input device index. Auto-selects headset if omitted.")
@click.option("--output-device", "-o", type=int, default=None,
              help="Output device index for TTS. Auto-selects headset if omitted.")
@click.option("--seconds", "-s", type=float, default=4.0,
              help="Seconds to record before transcribing.")
@click.option("--stt-engine", type=click.Choice(["stub", "whisper"]), default="whisper",
              help="STT backend.")
@click.option("--stt-model", type=str, default="tiny",
              help="Whisper model name.")
@click.option("--brain", type=click.Choice(["echo", "http", "hermes", "hermes-cli"]), default="echo",
              help="Use echo for offline loop test, http/hermes for raw OpenAI-compatible core, or hermes-cli for real Hermes Agent.")
@click.option("--no-tts", is_flag=True, help="Do not speak the response; print only.")
def listen_once(
    input_device: int | None,
    output_device: int | None,
    seconds: float,
    stt_engine: str,
    stt_model: str,
    brain: str,
    no_tts: bool,
):
    """Record once, transcribe, send to brain, and speak the response."""
    _setup_logging(verbose=True)
    base = Config.from_env()
    dev_mgr = AudioDeviceManager()
    dev_mgr.refresh()
    selected_input = dev_mgr.select_input(input_device)
    selected_output = dev_mgr.select_output(output_device)
    config = base.replace(
        input_device_index=selected_input.index,
        tts_output_device_index=selected_output.index,
        stt_engine=stt_engine,
        stt_model=stt_model,
        tts_playback_enabled=not no_tts,
    )

    click.echo(f"Input:  {selected_input.display_name}")
    click.echo(f"Output: {selected_output.display_name}")
    click.echo(f"Recording {seconds:.1f}s. Speak now, sir...")

    tracker = LatencyTracker()
    turn = tracker.new_turn()
    logger = LatencyLogger(config.log_dir)
    recorder = AudioRecorder(config)
    if brain == "echo":
        bridge = EchoBridge(chunk_size=12)
    elif brain == "hermes-cli":
        bridge = HermesCliBridge()
    else:
        bridge = HermesBridge(config)
    tts = TTSEngine(config)

    audio_path = config.cache_dir / "recordings" / "listen_once.wav"
    try:
        recorder.record_seconds(seconds, audio_path, selected_input.index, latency=turn)
        click.echo(f"Recorded: {audio_path}")

        stt = STTEngine(config)
        result = stt.transcribe_file(audio_path)
        transcript = result.text
        click.echo(f"Transcript: {transcript!r}")

        if not transcript:
            response = "I did not catch anything, sir. A silent performance, but technically flawless."
        else:
            if brain == "echo":
                chunks = []
                async def _echo():
                    async for chunk in bridge.stream(transcript):
                        chunks.append(chunk)
                asyncio.run(_echo())
                response = "".join(chunks)
            else:
                response = asyncio.run(bridge.send(transcript, latency=turn))
        click.echo(f"JARVIS: {response}")
        asyncio.run(tts.speak(response, latency=turn))
    finally:
        recorder.close()
        asyncio.run(tts.close())
        if hasattr(bridge, "close"):
            asyncio.run(bridge.close())
        tracker.finish_turn(turn)
        logger.log_turn(turn, turn_index=0)
        logger.close()
        click.echo(f"Latency log: {logger.current_file}")
        click.echo(f"TtFA: {turn.total_ttfa_ms:.0f}ms")


@main.command()
@click.option("--audio", "audio_path", type=click.Path(path_type=Path), required=True,
              help="Audio file to transcribe.")
@click.option("--engine", type=click.Choice(["stub", "whisper"]), default="whisper",
              help="STT backend to use.")
@click.option("--model", type=str, default="tiny",
              help="Whisper model name when --engine whisper.")
def transcribe_test(audio_path: Path, engine: str, model: str):
    """Transcribe a recorded WAV file."""
    _setup_logging(verbose=True)
    config = Config.from_env().replace(stt_engine=engine, stt_model=model)
    stt = STTEngine(config)
    result = stt.transcribe_file(audio_path)
    click.echo(f"Backend: {result.backend}")
    if result.language:
        click.echo(f"Language: {result.language}")
    click.echo(f"Transcript: {result.text}")


@main.command()
@click.option("--url", type=str, default=None,
              help="Hermes bridge URL.")
@click.option("--text", type=str, default="Hello, JARVIS. This is a bridge test.",
              help="Test transcript to send.")
def test_bridge(url: str | None, text: str):
    """Test the Hermes bridge by sending a transcript."""
    _setup_logging(verbose=True)
    config = Config.from_env()
    if url:
        config = config.replace(hermes_bridge_url=url)
    bridge = HermesBridge(config)

    async def _test():
        click.echo(f"Sending to {config.hermes_bridge_url}: '{text}'")
        response = await bridge.send(text)
        click.echo(f"Response: {response}")
        await bridge.close()

    asyncio.run(_test())


if __name__ == "__main__":
    main()
