"""Tests for always-on voice activity detection."""

from __future__ import annotations

from jarvis_voice_shell.vad import EnergyVAD, VADState, rms_int16


def frame(value: int, samples: int = 160) -> bytes:
    return b"".join(int(value).to_bytes(2, "little", signed=True) for _ in range(samples))


def test_rms_int16_distinguishes_quiet_and_loud_frames():
    assert rms_int16(frame(0)) == 0
    assert rms_int16(frame(3000)) > 2500


def test_energy_vad_emits_speech_segment_after_trailing_silence():
    vad = EnergyVAD(energy_threshold=500, start_frames=2, end_silence_frames=2, pre_roll_frames=1)

    outputs = []
    for chunk in [frame(0), frame(1000), frame(1200), frame(1300), frame(0), frame(0)]:
        segment = vad.process(chunk)
        if segment is not None:
            outputs.append(segment)

    assert len(outputs) == 1
    assert outputs[0].startswith(frame(0))
    assert frame(1300) in outputs[0]
    assert vad.state == VADState.IDLE


def test_energy_vad_ignores_short_noise_spike():
    vad = EnergyVAD(energy_threshold=500, start_frames=2, end_silence_frames=2, pre_roll_frames=1)

    for chunk in [frame(0), frame(1200), frame(0), frame(0)]:
        assert vad.process(chunk) is None

    assert vad.state == VADState.IDLE
