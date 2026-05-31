"""Push-to-talk controller abstractions for JARVIS Voice Shell.

The keyboard package is optional because Windows/global hotkeys can require
permissions or fail in non-interactive shells. The typed controller remains the
safe fallback and the real keyboard controller presents the same event shape.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PTTEvent(Enum):
    """Events emitted by PTT controllers."""

    PRESS = "press"
    RELEASE = "release"
    INTERRUPT = "interrupt"
    TEXT = "text"


@dataclass(frozen=True)
class PTTInputEvent:
    """One PTT or typed-input event."""

    kind: PTTEvent
    text: str | None = None


class TypedPTTController:
    """Typed stdin fallback when real global hotkeys are unavailable."""

    is_real_hotkey = False
    status = "typed fallback mode"

    @staticmethod
    def make_text_event(text: str) -> PTTInputEvent:
        return PTTInputEvent(PTTEvent.TEXT, text=text)


class KeyboardPTTController:
    """Keyboard-backed push-to-talk controller using the optional keyboard package."""

    is_real_hotkey = True

    def __init__(self, ptt_key: str, keyboard_module: Any | None = None):
        self.ptt_key = ptt_key
        self._keyboard = keyboard_module
        self._queue: asyncio.Queue[PTTInputEvent] = asyncio.Queue()
        self._registered = False
        self._press_hook: Any | None = None
        self._release_hook: Any | None = None

    @property
    def status(self) -> str:
        return f"keyboard hotkey mode ({self.ptt_key})"

    def _load_keyboard(self) -> Any:
        if self._keyboard is not None:
            return self._keyboard
        try:
            import keyboard  # type: ignore
        except ImportError as exc:
            raise RuntimeError("keyboard package is not installed") from exc
        return keyboard

    def start(self) -> None:
        """Register press/release hooks."""
        keyboard = self._load_keyboard()
        if self._registered:
            return
        self._press_hook = keyboard.on_press_key(
            self.ptt_key,
            lambda _event: self._queue.put_nowait(PTTInputEvent(PTTEvent.PRESS)),
            suppress=False,
        )
        self._release_hook = keyboard.on_release_key(
            self.ptt_key,
            lambda _event: self._queue.put_nowait(PTTInputEvent(PTTEvent.RELEASE)),
            suppress=False,
        )
        self._registered = True

    async def next_event(self) -> PTTInputEvent:
        return await self._queue.get()

    def stop(self) -> None:
        if not self._registered:
            return
        keyboard = self._load_keyboard()
        for hook in (self._press_hook, self._release_hook):
            if hook is not None:
                try:
                    keyboard.unhook(hook)
                except Exception:
                    pass
        self._registered = False
