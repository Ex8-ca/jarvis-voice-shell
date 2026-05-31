"""CLI behavior tests."""

from click.testing import CliRunner

from jarvis_voice_shell.cli import main


def test_run_help_exposes_always_on_input_mode():
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0
    assert "always-on" in result.output


def test_run_help_exposes_tts_rate_override():
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0
    assert "--tts-rate" in result.output
