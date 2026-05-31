"""Tests for push-to-talk controller abstractions."""

import pytest

from jarvis_voice_shell.ptt import PTTEvent, TypedPTTController


class TestPTTEvent:
    def test_event_names_are_stable(self):
        assert PTTEvent.PRESS.value == "press"
        assert PTTEvent.RELEASE.value == "release"
        assert PTTEvent.INTERRUPT.value == "interrupt"
        assert PTTEvent.TEXT.value == "text"


class TestTypedPTTController:
    def test_typed_controller_reports_no_real_hotkey(self):
        controller = TypedPTTController()
        assert controller.is_real_hotkey is False
        assert "typed" in controller.status.lower()

    def test_text_event_carries_payload(self):
        event = TypedPTTController.make_text_event("hello")
        assert event.kind == PTTEvent.TEXT
        assert event.text == "hello"


class FakeKeyboard:
    def __init__(self):
        self.press_key = None
        self.release_key = None
        self.unhooked = []

    def on_press_key(self, key, callback, suppress=False):
        self.press_key = key
        self.press_callback = callback
        return "press-hook"

    def on_release_key(self, key, callback, suppress=False):
        self.release_key = key
        self.release_callback = callback
        return "release-hook"

    def unhook(self, hook):
        self.unhooked.append(hook)


@pytest.mark.asyncio
async def test_keyboard_controller_emits_press_and_release_events():
    from jarvis_voice_shell.ptt import KeyboardPTTController

    fake = FakeKeyboard()
    controller = KeyboardPTTController("ctrl+shift+space", keyboard_module=fake)
    controller.start()

    fake.press_callback(None)
    fake.release_callback(None)

    assert (await controller.next_event()).kind == PTTEvent.PRESS
    assert (await controller.next_event()).kind == PTTEvent.RELEASE
    assert fake.press_key == "ctrl+shift+space"
    assert fake.release_key == "ctrl+shift+space"

    controller.stop()
    assert fake.unhooked == ["press-hook", "release-hook"]
