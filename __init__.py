"""hermes-voice plugin — voice interface for Hermes Agent.

When a user installs this plugin via::

    hermes plugins install Ex8-ca/jarvis-voice-shell

The FastAPI voice server starts automatically (if not already running)
and voice tools are registered with Hermes.

Users interact with the voice UI at http://<host>:8989/ (or via Tailscale).
The plugin also registers a ``hermes_voice_status`` tool and
``/hermes-voice`` slash command for management.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("hermes-voice.plugin")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PORT = int(os.environ.get("HERMES_VOICE_PORT", "8989"))
_SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

# Module-level process handle
_server_process: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _get_plugin_dir() -> Path:
    """Return the plugin directory (where this __init__.py lives)."""
    return Path(__file__).resolve().parent


def _port_in_use(port: int) -> bool:
    """Check if a TCP port is already listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _is_server_responding(port: int) -> bool:
    """Check if the voice server is actually responding to HTTP."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def start_server(*, port: Optional[int] = None, quiet: bool = True) -> bool:
    """Start the hermes_voice FastAPI server as a subprocess.

    Returns True if the server is running (or was already running).
    """
    global _server_process

    port = port or DEFAULT_PORT

    if _port_in_use(port) and _is_server_responding(port):
        logger.info("hermes-voice server already running on port %d", port)
        return True

    plugin_dir = _get_plugin_dir()

    # Find a python that has the required packages
    python = sys.executable or "python3"

    cmd = [
        python, "-m", "uvicorn",
        "hermes_voice.gateway:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "info",
    ]

    env = os.environ.copy()
    # Ensure the plugin dir is on PYTHONPATH so `hermes_voice` subpackage
    # is importable (plugin dir is already in __path__, but subprocess
    # doesn't inherit that).
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{plugin_dir}:{pp}" if pp else str(plugin_dir)

    logger.info("Starting hermes-voice server: %s", " ".join(cmd[:4]))

    try:
        if quiet:
            _server_process = subprocess.Popen(
                cmd,
                cwd=str(plugin_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            _server_process = subprocess.Popen(
                cmd,
                cwd=str(plugin_dir),
                env=env,
            )
        atexit.register(_stop_server)
        logger.info("hermes-voice server started (PID %d, port %d)", _server_process.pid, port)

        # Wait a moment for startup
        import time
        time.sleep(2)
        return _is_server_responding(port)

    except Exception as e:
        logger.error("Failed to start hermes-voice server: %s", e)
        return False


def _stop_server() -> None:
    """Stop the hermes_voice FastAPI server."""
    global _server_process
    if _server_process is None:
        return
    try:
        _server_process.terminate()
        _server_process.wait(timeout=5)
        logger.info("hermes-voice server stopped")
    except subprocess.TimeoutExpired:
        _server_process.kill()
        _server_process.wait()
        logger.info("hermes-voice server killed")
    except Exception as e:
        logger.error("Error stopping hermes-voice server: %s", e)
    finally:
        _server_process = None


def stop_server() -> bool:
    """Public stop — callable from slash command."""
    _stop_server()
    return not _port_in_use(DEFAULT_PORT)


def restart_server(*, port: Optional[int] = None) -> bool:
    """Restart the hermes_voice FastAPI server."""
    stop_server()
    import time
    time.sleep(1)
    return start_server(port=port)


def get_server_status() -> dict:
    """Return the current voice server status."""
    port = DEFAULT_PORT
    return {
        "running": _is_server_responding(port),
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "pid": _server_process.pid if _server_process else None,
    }


# ---------------------------------------------------------------------------
# Tool: hermes_voice_status
# ---------------------------------------------------------------------------

HERMES_VOICE_STATUS_SCHEMA = {
    "name": "hermes_voice_status",
    "description": (
        "Check the status of the Hermes Voice server. "
        "Returns whether the server is running, its URL, and PID. "
        "Use this before attempting voice interactions."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def _handle_voice_status(args: dict, **kwargs) -> str:
    """Handler for the hermes_voice_status tool."""
    return json.dumps(get_server_status())


# ---------------------------------------------------------------------------
# Slash command: /hermes-voice
# ---------------------------------------------------------------------------

_VOICE_CMD_HELP = """\
/hermes-voice — Manage the Hermes Voice server

Subcommands:
  start [port]     — Start the voice server (default port: 8989)
  stop             — Stop the voice server
  restart [port]   — Restart the voice server
  status           — Show server status
"""


def _handle_voice_command(raw_args: str) -> Optional[str]:
    """Handle /hermes-voice slash command."""
    argv = raw_args.strip().split()
    sub = argv[0] if argv else "status"

    if sub == "start":
        port = int(argv[1]) if len(argv) > 1 else DEFAULT_PORT
        if start_server(port=port):
            return f"Hermes Voice server started at http://127.0.0.1:{port}"
        return "Failed to start Hermes Voice server — check logs"

    if sub == "stop":
        if stop_server():
            return "Hermes Voice server stopped"
        return "Failed to stop Hermes Voice server"

    if sub == "restart":
        port = int(argv[1]) if len(argv) > 1 else DEFAULT_PORT
        if restart_server(port=port):
            return f"Hermes Voice server restarted at http://127.0.0.1:{port}"
        return "Failed to restart Hermes Voice server"

    if sub == "status":
        status = get_server_status()
        lines = ["Hermes Voice Server:"]
        lines.append(f"  Running: {'Yes' if status['running'] else 'No'}")
        lines.append(f"  URL: {status['url']}")
        lines.append(f"  PID: {status['pid'] or 'N/A'}")
        return "\n".join(lines)

    return _VOICE_CMD_HELP


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register hermes-voice with Hermes. Called once by the plugin loader."""

    # 1. Register the status tool
    ctx.register_tool(
        name="hermes_voice_status",
        toolset="hermes_voice",
        schema=HERMES_VOICE_STATUS_SCHEMA,
        handler=lambda args, **kw: _handle_voice_status(args, **kw),
    )

    # 2. Register the management slash command
    ctx.register_command(
        "hermes-voice",
        handler=_handle_voice_command,
        description="Manage the Hermes Voice server (start/stop/restart/status).",
        args_hint="<start|stop|restart|status> [port]",
    )

    # 3. Auto-start the voice server (non-blocking, best-effort)
    #    If the server is already running, this is a no-op.
    #    If it fails to start, tools are still registered — user can
    #    start manually with /hermes-voice start.
    try:
        start_server(quiet=True)
    except Exception as e:
        logger.warning("hermes-voice server auto-start failed (start manually with /hermes-voice start): %s", e)

    logger.info("hermes-voice plugin registered (tool + slash command + auto-start)")
