"""Tests for interruptible speech queue."""

from __future__ import annotations

import asyncio

import pytest

from jarvis_voice_shell.speech_queue import InterruptibleSpeechQueue, split_for_speech


def test_split_for_speech_preserves_sentence_chunks():
    assert split_for_speech("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_split_for_speech_batches_short_fragments():
    assert split_for_speech("Systems nominal, sir") == ["Systems nominal, sir"]


@pytest.mark.asyncio
async def test_speech_queue_speaks_chunks_in_order():
    spoken: list[str] = []

    async def speak(text: str):
        spoken.append(text)

    queue = InterruptibleSpeechQueue(speak=speak)
    await queue.start()
    await queue.enqueue("One. Two.")
    await queue.join()
    await queue.close()

    assert spoken == ["One.", "Two."]


@pytest.mark.asyncio
async def test_interrupt_clears_pending_chunks_and_cancels_current_speech():
    started = asyncio.Event()
    cancelled = False
    spoken: list[str] = []

    async def speak(text: str):
        nonlocal cancelled
        spoken.append(text)
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    queue = InterruptibleSpeechQueue(speak=speak)
    await queue.start()
    await queue.enqueue("One. Two.")
    await started.wait()

    await queue.interrupt()
    await queue.close()

    assert cancelled is True
    assert spoken == ["One."]
