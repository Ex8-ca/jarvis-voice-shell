"""Energy-based voice activity detection for always-on JARVIS mode."""

from __future__ import annotations

import math
from collections import deque
from enum import Enum


class VADState(str, Enum):
    """VAD segmenter state."""

    IDLE = "idle"
    PRIMED = "primed"
    SPEAKING = "speaking"


def rms_int16(frame: bytes) -> int:
    """Return RMS amplitude for little-endian signed 16-bit mono PCM."""
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
    """Simple energy VAD that emits complete speech segments.

    A segment begins after ``start_frames`` consecutive loud frames and ends
    after ``end_silence_frames`` consecutive quiet frames. ``pre_roll_frames``
    are prepended so initial consonants are less likely to be clipped.
    """

    def __init__(
        self,
        energy_threshold: int = 500,
        start_frames: int = 3,
        end_silence_frames: int = 11,
        pre_roll_frames: int = 5,
    ):
        self.energy_threshold = int(energy_threshold)
        self.start_frames = max(1, int(start_frames))
        self.end_silence_frames = max(1, int(end_silence_frames))
        self.pre_roll_frames = max(0, int(pre_roll_frames))
        self.state = VADState.IDLE
        self._pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._primed: list[bytes] = []
        self._segment: list[bytes] = []
        self._loud_count = 0
        self._quiet_count = 0

    def process(self, frame: bytes) -> bytes | None:
        """Process one PCM frame; return a segment when speech ends."""
        loud = rms_int16(frame) >= self.energy_threshold

        if self.state == VADState.IDLE:
            if loud:
                self.state = VADState.PRIMED
                self._primed = [frame]
                self._loud_count = 1
            else:
                self._pre_roll.append(frame)
            return None

        if self.state == VADState.PRIMED:
            if loud:
                self._primed.append(frame)
                self._loud_count += 1
                if self._loud_count >= self.start_frames:
                    self.state = VADState.SPEAKING
                    self._segment = list(self._pre_roll) + self._primed
                    self._quiet_count = 0
                    self._pre_roll.clear()
                    self._primed = []
            else:
                for old in self._primed:
                    self._pre_roll.append(old)
                self._pre_roll.append(frame)
                self._primed = []
                self._loud_count = 0
                self.state = VADState.IDLE
            return None

        # SPEAKING
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
        """Reset detector state."""
        self.state = VADState.IDLE
        self._pre_roll.clear()
        self._primed = []
        self._segment = []
        self._loud_count = 0
        self._quiet_count = 0
