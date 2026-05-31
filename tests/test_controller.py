"""Tests for full-duplex conversation controller."""

from __future__ import annotations

import asyncio

import pytest

from jarvis_voice_shell.controller import ConversationController, ConversationState


@pytest.mark.asyncio
async def test_start_turn_advances_turn_id_and_cancels_previous_turn():
    controller = ConversationController()

    first = controller.start_turn()
    second = controller.start_turn()

    assert first.turn_id == 1
    assert second.turn_id == 2
    assert first.cancel_event.is_set()
    assert not second.cancel_event.is_set()
    assert controller.current_turn_id == 2


@pytest.mark.asyncio
async def test_interrupt_cancels_active_task_and_calls_cancel_playback():
    cancelled = False
    playback_cancelled = False

    async def worker():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def cancel_playback():
        nonlocal playback_cancelled
        playback_cancelled = True

    controller = ConversationController(cancel_playback=cancel_playback)
    turn = controller.start_turn()
    task = asyncio.create_task(worker())
    controller.track_task(task, turn.turn_id)
    await asyncio.sleep(0)

    await controller.interrupt()

    assert playback_cancelled is True
    assert cancelled is True
    assert turn.cancel_event.is_set()
    assert controller.state == ConversationState.INTERRUPTED


@pytest.mark.asyncio
async def test_ignore_stale_turns_after_new_turn_starts():
    controller = ConversationController()
    first = controller.start_turn()
    second = controller.start_turn()

    assert controller.is_current(first.turn_id) is False
    assert controller.is_current(second.turn_id) is True
