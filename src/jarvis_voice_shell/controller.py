"""Conversation turn state for full-duplex JARVIS behavior.

The controller owns cancellation and turn freshness. Audio capture, Hermes
thinking, and TTS playback can run as separate tasks while checking this shared
state so a newer user utterance can immediately supersede older work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable


class ConversationState(str, Enum):
    """High-level state labels for the realtime voice shell."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class TurnContext:
    """Cancellation-aware context for one user turn."""

    turn_id: int
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


CancelPlayback = Callable[[], Awaitable[None]]


async def _noop_cancel_playback() -> None:
    return None


class ConversationController:
    """Owns current turn id, state, and cancellation of stale tasks."""

    def __init__(self, cancel_playback: CancelPlayback | None = None):
        self._next_turn_id = 0
        self._current: TurnContext | None = None
        self._tasks: dict[int, set[asyncio.Task]] = {}
        self._cancel_playback = cancel_playback or _noop_cancel_playback
        self.state = ConversationState.IDLE

    @property
    def current_turn_id(self) -> int | None:
        return self._current.turn_id if self._current else None

    def start_turn(self) -> TurnContext:
        """Start a new turn and mark the previous turn stale/cancelled."""
        if self._current is not None:
            self._current.cancel_event.set()
            self._cancel_tasks(self._current.turn_id)
        self._next_turn_id += 1
        self._current = TurnContext(self._next_turn_id)
        self.state = ConversationState.LISTENING
        return self._current

    def track_task(self, task: asyncio.Task, turn_id: int | None = None) -> None:
        """Track a task so interrupts can cancel it."""
        tid = turn_id if turn_id is not None else self.current_turn_id
        if tid is None:
            return
        self._tasks.setdefault(tid, set()).add(task)
        task.add_done_callback(lambda done, tid=tid: self._tasks.get(tid, set()).discard(done))

    def is_current(self, turn_id: int) -> bool:
        """Return True if ``turn_id`` still owns the conversation."""
        return self._current is not None and self._current.turn_id == turn_id and not self._current.cancel_event.is_set()

    async def interrupt(self) -> None:
        """Stop playback and cancel all active work for the current turn."""
        if self._current is not None:
            self._current.cancel_event.set()
            tasks = list(self._tasks.get(self._current.turn_id, set()))
        else:
            tasks = []
        self.state = ConversationState.INTERRUPTED
        await self._cancel_playback()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def set_state(self, state: ConversationState, turn_id: int | None = None) -> None:
        """Update state only when the turn is still current."""
        if turn_id is None or self.is_current(turn_id):
            self.state = state

    def _cancel_tasks(self, turn_id: int) -> None:
        for task in list(self._tasks.get(turn_id, set())):
            if not task.done():
                task.cancel()
