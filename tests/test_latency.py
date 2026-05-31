"""Tests for latency.py."""

import json

import pytest

from jarvis_voice_shell.latency import LatencyLogger, LatencyTracker, TurnLatency


class TestTurnLatency:
    """Per-turn latency object."""

    def test_stt_ms_positive(self):
        turn = TurnLatency(stt_start=0.0, stt_end=0.5)
        assert turn.stt_ms == 500.0

    def test_bridge_ttfb_ms(self):
        turn = TurnLatency(
            bridge_start=0.0, bridge_first_token=0.250,
        )
        assert turn.bridge_ttfb_ms == 250.0

    def test_total_ttfa_ms(self):
        turn = TurnLatency(
            turn_start=0.0, tts_first_audio=1.5,
        )
        assert turn.total_ttfa_ms == 1500.0

    def test_all_zero_by_default(self):
        turn = TurnLatency()
        assert turn.stt_ms == 0.0
        assert turn.bridge_ttfb_ms == 0.0
        assert turn.total_ttfa_ms == 0.0


class TestLatencyTracker:
    """Multi-turn tracker."""

    def test_empty_summary(self):
        tracker = LatencyTracker()
        assert "No turns" in tracker.summary()

    def test_single_turn(self):
        tracker = LatencyTracker()
        turn = tracker.new_turn()
        # Override turn_start to match our synthetic timeline
        turn.turn_start = 1000.0
        turn.stt_start = 1000.0
        turn.stt_end = 1000.5
        turn.bridge_start = 1000.5
        turn.bridge_first_token = 1000.750
        turn.bridge_last_token = 1002.0
        turn.tts_start = 1002.0
        turn.tts_first_audio = 1002.3
        turn.tts_end = 1004.0
        tracker.finish_turn(turn)

        assert turn.stt_ms == pytest.approx(500.0)
        assert turn.bridge_ttfb_ms == pytest.approx(250.0)
        assert turn.bridge_total_ms == pytest.approx(1500.0)
        assert turn.tts_latency_ms == pytest.approx(300.0)
        assert turn.total_ttfa_ms == pytest.approx(2300.0)

    def test_avg_ttfa_empty(self):
        tracker = LatencyTracker()
        assert tracker.avg_ttfa_ms == 0.0

    def test_finish_turn_stamps_wall_clock(self):
        tracker = LatencyTracker()
        turn = tracker.new_turn()
        assert turn.turn_end == 0.0
        tracker.finish_turn(turn)
        assert turn.turn_end > 0.0


class TestLatencyLogger:
    """JSONL event logging for latency metrics."""

    def make_turn(self, **overrides) -> TurnLatency:
        """Helper: create a synthetic turn with known timestamps."""
        defaults = {
            "turn_start": 1000.0,
            "stt_start": 1000.0,
            "stt_end": 1000.5,
            "bridge_start": 1000.5,
            "bridge_first_token": 1000.75,
            "bridge_last_token": 1002.0,
            "tts_start": 1002.0,
            "tts_first_audio": 1002.3,
            "tts_end": 1004.0,
            "turn_end": 1004.0,
        }
        defaults.update(overrides)
        t = TurnLatency()
        for k, v in defaults.items():
            setattr(t, k, v)
        return t

    def test_log_turn_writes_jsonl_line(self, tmp_path):
        log_dir = tmp_path / "latency_logs"
        logger = LatencyLogger(log_dir=log_dir)
        turn = self.make_turn()
        logger.log_turn(turn, turn_index=0)

        logs = list(log_dir.glob("*.jsonl"))
        assert len(logs) == 1, f"Expected one JSONL file in {log_dir}, found {logs}"

        lines = logs[0].read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["turn_index"] == 0
        assert record["stt_ms"] == pytest.approx(500.0)
        assert record["bridge_ttfb_ms"] == pytest.approx(250.0)
        assert record["bridge_total_ms"] == pytest.approx(1500.0)
        assert record["tts_latency_ms"] == pytest.approx(300.0)
        assert record["tts_total_ms"] == pytest.approx(2000.0)
        assert record["total_ttfa_ms"] == pytest.approx(2300.0)
        assert "timestamp" in record
        assert "turn_start" in record
        assert "turn_end" in record

    def test_log_multiple_turns_appends(self, tmp_path):
        log_dir = tmp_path / "latency_logs"
        logger = LatencyLogger(log_dir=log_dir)

        logger.log_turn(self.make_turn(), turn_index=0)
        logger.log_turn(self.make_turn(turn_start=2000.0, tts_first_audio=2002.0,
                                        turn_end=2004.0), turn_index=1)

        logs = list(log_dir.glob("*.jsonl"))
        lines = logs[0].read_text().strip().splitlines()
        assert len(lines) == 2
        r0 = json.loads(lines[0])
        r1 = json.loads(lines[1])
        assert r0["turn_index"] == 0
        assert r1["turn_index"] == 1
        assert r0["turn_start"] == pytest.approx(1000.0)
        assert r1["turn_start"] == pytest.approx(2000.0)

    def test_log_dir_created_if_missing(self, tmp_path):
        log_dir = tmp_path / "nonexistent" / "latency"
        assert not log_dir.exists()
        logger = LatencyLogger(log_dir=log_dir)
        logger.log_turn(self.make_turn(), turn_index=0)
        assert log_dir.exists()
        assert len(list(log_dir.glob("*.jsonl"))) == 1

    def test_flush_does_not_raise(self, tmp_path):
        log_dir = tmp_path / "latency_logs"
        logger = LatencyLogger(log_dir=log_dir)
        logger.log_turn(self.make_turn(), turn_index=0)
        logger.flush()  # should not raise

    def test_flush_no_file_noop(self, tmp_path):
        logger = LatencyLogger(log_dir=tmp_path)
        logger.flush()  # should not raise

    def test_close_closes_open_file(self, tmp_path):
        logger = LatencyLogger(log_dir=tmp_path)
        logger.log_turn(self.make_turn(), turn_index=0)
        path = logger.current_file
        logger.close()
        assert path is not None
        assert path.exists()
        assert logger.current_file == path

    def test_log_filename_is_timestamped(self, tmp_path):
        log_dir = tmp_path / "latency_logs"
        logger = LatencyLogger(log_dir=log_dir)
        logger.log_turn(self.make_turn(), turn_index=0)
        logs = list(log_dir.glob("*.jsonl"))
        assert logs[0].stem.startswith("jarvis_latency_")

    def test_jsonl_lines_are_valid_json(self, tmp_path):
        log_dir = tmp_path / "latency_logs"
        logger = LatencyLogger(log_dir=log_dir)
        turn = self.make_turn()
        logger.log_turn(turn, turn_index=0)

        logs = list(log_dir.glob("*.jsonl"))
        for line in logs[0].read_text().strip().splitlines():
            record = json.loads(line)
            # Verify all expected keys are present
            for key in ("timestamp", "turn_index", "stt_ms", "bridge_ttfb_ms",
                         "bridge_total_ms", "tts_latency_ms", "tts_total_ms",
                         "total_ttfa_ms"):
                assert key in record, f"Missing key '{key}' in JSONL record"

    def test_logger_with_default_path_uses_cache_dir(self, tmp_path, monkeypatch):
        """When no log_dir given, uses the default cache/log path."""
        # Simulate Windows LOCALAPPDATA
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        import os
        if os.name != "nt":
            # On non-Windows, set HOME to tmp_path so cache dir goes there
            monkeypatch.setattr(
                "jarvis_voice_shell.latency._default_log_dir",
                lambda: tmp_path / ".cache" / "jarvis-voice-shell" / "latency_logs",
            )

    def test_logger_survives_nonexistent_path(self, tmp_path):
        """Logging to a path where parent doesn't exist should create it."""
        deep = tmp_path / "a" / "b" / "c"
        logger = LatencyLogger(log_dir=deep)
        logger.log_turn(self.make_turn(), turn_index=0)
        assert deep.exists()


def test_discard_turn_removes_aborted_turn_from_summary():
    from jarvis_voice_shell.latency import LatencyTracker

    tracker = LatencyTracker()
    turn = tracker.new_turn()
    tracker.discard_turn(turn)

    assert tracker.turns == []
    assert tracker.summary() == "No turns recorded."
