"""Latency tracking for the JARVIS voice pipeline.

Measures four key phases:
    1.  STT latency    — mic-off to final transcript
    2.  Bridge TTFB    — time-to-first-byte from Hermes
    3.  Bridge total   — full response streaming duration
    4.  TTS latency    — first text token to first audio sample played

All timestamps are perf_counter (monotonic, high resolution).

Also provides LatencyLogger for durable JSONL event logging to disk.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_log_dir() -> Path:
    """Platform-appropriate default for latency logs."""
    import os
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "jarvis-voice-shell" / "latency_logs"
    return Path.home() / ".cache" / "jarvis-voice-shell" / "latency_logs"


@dataclass
class TurnLatency:
    """Per-turn latency measurements."""

    # STT phase
    stt_start: float = 0.0
    stt_end: float = 0.0

    # Bridge phase (streaming-aware)
    bridge_start: float = 0.0
    bridge_first_token: float = 0.0    # TTFB
    bridge_last_token: float = 0.0

    # TTS phase
    tts_start: float = 0.0
    tts_first_audio: float = 0.0       # first audio sample played
    tts_end: float = 0.0

    # Wall clock
    turn_start: float = 0.0
    turn_end: float = 0.0

    @property
    def stt_ms(self) -> float:
        """Speech-to-text duration in milliseconds."""
        return (self.stt_end - self.stt_start) * 1000

    @property
    def bridge_ttfb_ms(self) -> float:
        """Time-to-first-byte from Hermes in milliseconds."""
        return (self.bridge_first_token - self.bridge_start) * 1000

    @property
    def bridge_total_ms(self) -> float:
        """Full bridge streaming duration in milliseconds."""
        return (self.bridge_last_token - self.bridge_start) * 1000

    @property
    def tts_latency_ms(self) -> float:
        """TTS first-audio latency in milliseconds."""
        return (self.tts_first_audio - self.tts_start) * 1000

    @property
    def tts_total_ms(self) -> float:
        """Total TTS duration in milliseconds."""
        return (self.tts_end - self.tts_start) * 1000

    @property
    def total_ttfa_ms(self) -> float:
        """Time-to-first-audio — push-to-talk release to first audio sample."""
        return (self.tts_first_audio - self.turn_start) * 1000


@dataclass
class LatencyTracker:
    """Collects latency metrics across turns."""

    turns: list[TurnLatency] = field(default_factory=list)

    def new_turn(self) -> TurnLatency:
        """Start a new turn, recording wall-clock start."""
        turn = TurnLatency(turn_start=time.perf_counter())
        self.turns.append(turn)
        return turn

    def finish_turn(self, turn: TurnLatency) -> None:
        """Stamp wall-clock end on a completed turn."""
        turn.turn_end = time.perf_counter()

    def discard_turn(self, turn: TurnLatency) -> None:
        """Remove an aborted turn so partial metrics do not poison summaries."""
        try:
            self.turns.remove(turn)
        except ValueError:
            pass

    @property
    def avg_ttfa_ms(self) -> float:
        """Average time-to-first-audio across all turns."""
        if not self.turns:
            return 0.0
        return sum(t.total_ttfa_ms for t in self.turns) / len(self.turns)

    def summary(self) -> str:
        """Human-readable summary of latency stats."""
        if not self.turns:
            return "No turns recorded."
        n = len(self.turns)
        avg_stt = sum(t.stt_ms for t in self.turns) / n
        avg_ttfb = sum(t.bridge_ttfb_ms for t in self.turns) / n
        avg_bridge = sum(t.bridge_total_ms for t in self.turns) / n
        avg_tts_lat = sum(t.tts_latency_ms for t in self.turns) / n
        avg_ttfa = sum(t.total_ttfa_ms for t in self.turns) / n
        return (
            f"Latency ({n} turn{'s' if n != 1 else ''}):\n"
            f"  STT:          {avg_stt:7.1f} ms\n"
            f"  Bridge TTFB:  {avg_ttfb:7.1f} ms\n"
            f"  Bridge total: {avg_bridge:7.1f} ms\n"
            f"  TTS latency:  {avg_tts_lat:7.1f} ms\n"
            f"  Total TtFA:   {avg_ttfa:7.1f} ms"
        )


class LatencyLogger:
    """Durable JSONL event logger for latency metrics.

    Writes one JSON line per turn to a timestamped file in log_dir.
    Thread-safe for sequential use (not concurrent).
    """

    def __init__(self, log_dir: Path | None = None):
        """Create a latency logger.

        Args:
            log_dir: Directory for JSONL log files.
                     Defaults to platform-appropriate cache location.
        """
        self._log_dir = log_dir if log_dir is not None else _default_log_dir()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file: object | None = None
        self._file_path: Path | None = None

    def _ensure_file(self) -> None:
        """Open the log file if not already open."""
        if self._file is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            self._file_path = self._log_dir / f"jarvis_latency_{timestamp}.jsonl"
            self._file = open(self._file_path, "a", encoding="utf-8")

    def log_turn(self, turn: TurnLatency, turn_index: int = 0) -> None:
        """Write one turn's latency metrics as a JSONL line.

        Args:
            turn: Completed TurnLatency with all phases stamped.
            turn_index: Zero-based turn counter for ordering.
        """
        self._ensure_file()

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "turn_index": turn_index,
            "turn_start": turn.turn_start,
            "turn_end": turn.turn_end,
            "stt_ms": round(turn.stt_ms, 3),
            "bridge_ttfb_ms": round(turn.bridge_ttfb_ms, 3),
            "bridge_total_ms": round(turn.bridge_total_ms, 3),
            "tts_latency_ms": round(turn.tts_latency_ms, 3),
            "tts_total_ms": round(turn.tts_total_ms, 3),
            "total_ttfa_ms": round(turn.total_ttfa_ms, 3),
        }

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def flush(self) -> None:
        """Force flush the log file to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the current JSONL log file, if open."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    @property
    def log_dir(self) -> Path:
        """The directory where log files are written."""
        return self._log_dir

    @property
    def current_file(self) -> Path | None:
        """The currently-open log file, or None."""
        return self._file_path

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
