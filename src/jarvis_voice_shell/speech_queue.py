"""Interruptible TTS queue for full-duplex voice playback."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable


SpeakFn = Callable[[str], Awaitable[None]]

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


def split_for_speech(text: str) -> list[str]:
    """Split text into sentence-ish chunks for cancellable playback."""
    chunks = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text.strip())]
    return [chunk for chunk in chunks if chunk]


class InterruptibleSpeechQueue:
    """Queue sentence chunks and allow immediate interruption.

    The queue serializes TTS calls so a future streaming bridge can enqueue text
    as it arrives. ``interrupt()`` cancels the active speak call and drains any
    pending chunks, giving PTT/VAD a deterministic barge-in lever.
    """

    def __init__(self, speak: SpeakFn):
        self._speak = speak
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._current_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def enqueue(self, text: str) -> None:
        for chunk in split_for_speech(text):
            await self._queue.put(chunk)

    async def join(self) -> None:
        await self._queue.join()

    async def interrupt(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        await self.interrupt()
        if self._worker is not None and not self._worker.done():
            await self._queue.put(None)
            await self._worker
        self._worker = None

    async def _run(self) -> None:
        while True:
            chunk = await self._queue.get()
            try:
                if chunk is None:
                    return
                self._current_task = asyncio.create_task(self._speak(chunk))
                await self._current_task
            except asyncio.CancelledError:
                pass
            finally:
                self._current_task = None
                self._queue.task_done()
