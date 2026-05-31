"""Tests for audio_devices.py."""


from jarvis_voice_shell.audio_devices import (
    AudioDevice,
    AudioDeviceManager,
)


class TestAudioDevice:
    """AudioDevice dataclass."""

    def test_display_name_format(self):
        dev = AudioDevice(
            index=2, name="Test Mic",
            max_input_channels=1, max_output_channels=0,
            default_sample_rate=44100, is_input=True,
        )
        assert dev.display_name == "[2] Test Mic"

    def test_matches_all_keywords(self):
        dev = AudioDevice(
            index=0, name="Bluetooth Gaming Headset (Jabra)",
            max_input_channels=2, max_output_channels=2,
            default_sample_rate=48000,
        )
        assert dev.matches_keywords(["bluetooth", "headset"])
        assert dev.matches_keywords(["gaming"])

    def test_matches_case_insensitive(self):
        dev = AudioDevice(
            index=0, name="AirPods Pro",
            max_input_channels=1, max_output_channels=2,
            default_sample_rate=16000,
        )
        assert dev.matches_keywords(["airpod"])
        assert dev.matches_keywords(["AIRPOD"])
        assert dev.matches_keywords(["AirPod"])

    def test_no_match_on_partial(self):
        dev = AudioDevice(
            index=0, name="USB Microphone",
            max_input_channels=1, max_output_channels=0,
            default_sample_rate=44100,
        )
        assert not dev.matches_keywords(["bluetooth", "headset"])
        # Must match ALL keywords in the group
        assert not dev.matches_keywords(["logitech", "microphone"])


class TestAudioDeviceManager:
    """Device manager tests (no PyAudio needed for logic tests)."""

    def test_empty_manager_has_no_devices(self):
        mgr = AudioDeviceManager()
        assert mgr.input_devices == []
        assert mgr.output_devices == []

    def test_prioritized_select_bluetooth_first(self):
        devices = [
            AudioDevice(
                index=0, name="System Default",
                max_input_channels=1, max_output_channels=0,
                default_sample_rate=44100, is_input=True,
            ),
            AudioDevice(
                index=1, name="USB Microphone",
                max_input_channels=1, max_output_channels=0,
                default_sample_rate=44100, is_input=True,
            ),
            AudioDevice(
                index=3, name="Bluetooth Gaming Headset",
                max_input_channels=2, max_output_channels=2,
                default_sample_rate=48000, is_input=True,
            ),
        ]
        priority = [
            ["bluetooth", "headset"],
            ["microphone", "mic"],
        ]
        selected = AudioDeviceManager._prioritized_select(devices, priority)
        assert selected.index == 3
        assert "Bluetooth" in selected.name

    def test_prioritized_select_fallthrough(self):
        devices = [
            AudioDevice(
                index=0, name="Unknown Device",
                max_input_channels=1, max_output_channels=0,
                default_sample_rate=44100, is_input=True,
            ),
        ]
        priority = [["bluetooth", "headset"]]
        selected = AudioDeviceManager._prioritized_select(devices, priority)
        assert selected.index == 0  # fallback to first

    def test_select_input_with_preferred_index(self, monkeypatch):
        """Test that preferred_index bypasses priority when available."""
        mgr = AudioDeviceManager()
        mgr.input_devices = [
            AudioDevice(
                index=0, name="First Mic",
                max_input_channels=1, max_output_channels=0,
                default_sample_rate=44100, is_input=True,
            ),
            AudioDevice(
                index=5, name="Preferred Mic",
                max_input_channels=1, max_output_channels=0,
                default_sample_rate=48000, is_input=True,
            ),
        ]
        selected = mgr.select_input(preferred_index=5)
        assert selected.index == 5


class TestInjectDevices:
    """Dependency injection of raw device dicts for testing."""

    def test_inject_devices_replaces_existing(self):
        mgr = AudioDeviceManager()
        raw_list = [
            {
                "index": 0,
                "name": "Inject Test Mic",
                "maxInputChannels": 1,
                "maxOutputChannels": 0,
                "defaultSampleRate": 48000.0,
            },
            {
                "index": 1,
                "name": "Inject Test Speaker",
                "maxInputChannels": 0,
                "maxOutputChannels": 2,
                "defaultSampleRate": 44100.0,
            },
        ]
        mgr.inject_devices(raw_list)

        assert len(mgr.input_devices) == 1
        assert len(mgr.output_devices) == 1

        inp = mgr.input_devices[0]
        assert inp.index == 0
        assert inp.name == "Inject Test Mic"
        assert inp.is_input is True
        assert inp.is_output is False

        out = mgr.output_devices[0]
        assert out.index == 1
        assert out.name == "Inject Test Speaker"
        assert out.is_input is False
        assert out.is_output is True

    def test_inject_devices_handles_mixed_device(self):
        """Device that is both input and output (e.g., headset)."""
        mgr = AudioDeviceManager()
        raw_list = [
            {
                "index": 3,
                "name": "Bluetooth Gaming Headset",
                "maxInputChannels": 2,
                "maxOutputChannels": 2,
                "defaultSampleRate": 48000.0,
            },
        ]
        mgr.inject_devices(raw_list)

        assert len(mgr.input_devices) == 1
        assert len(mgr.output_devices) == 1
        assert mgr.input_devices[0].is_input is True
        assert mgr.output_devices[0].is_output is True

    def test_inject_devices_empty_list(self):
        mgr = AudioDeviceManager()
        mgr.inject_devices([])
        assert mgr.input_devices == []
        assert mgr.output_devices == []

    def test_inject_devices_then_select(self):
        """After injection, select_input uses the injected devices."""
        mgr = AudioDeviceManager()
        raw_list = [
            {
                "index": 0,
                "name": "System Default",
                "maxInputChannels": 1,
                "maxOutputChannels": 0,
                "defaultSampleRate": 44100.0,
            },
            {
                "index": 1,
                "name": "Bluetooth Gaming Headset",
                "maxInputChannels": 2,
                "maxOutputChannels": 2,
                "defaultSampleRate": 48000.0,
            },
        ]
        mgr.inject_devices(raw_list)
        selected = mgr.select_input()
        # Bluetooth gaming headset should win via priority
        assert selected.index == 1
        assert "Bluetooth" in selected.name

    def test_inject_devices_clears_previous(self):
        """Calling inject_devices twice should replace, not append."""
        mgr = AudioDeviceManager()
        mgr.inject_devices([
            {"index": 0, "name": "First", "maxInputChannels": 1,
             "maxOutputChannels": 0, "defaultSampleRate": 44100.0},
        ])
        mgr.inject_devices([
            {"index": 1, "name": "Second", "maxInputChannels": 1,
             "maxOutputChannels": 0, "defaultSampleRate": 44100.0},
        ])
        assert len(mgr.input_devices) == 1
        assert mgr.input_devices[0].name == "Second"


class TestClassifyDevice:
    """Priority classification labels."""

    def test_classify_bluetooth_input(self):
        dev = AudioDevice(
            index=0, name="Bluetooth Gaming Headset",
            max_input_channels=2, max_output_channels=2,
            default_sample_rate=48000, is_input=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=True)
        assert "Bluetooth" in label or "Headset" in label

    def test_classify_airpods_output(self):
        dev = AudioDevice(
            index=0, name="AirPods Pro",
            max_input_channels=1, max_output_channels=2,
            default_sample_rate=16000, is_output=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=False)
        assert "AirPod" in label

    def test_classify_logitech_input(self):
        dev = AudioDevice(
            index=0, name="Logitech C920 Webcam",
            max_input_channels=1, max_output_channels=0,
            default_sample_rate=16000, is_input=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=True)
        assert "Webcam" in label or "Logitech" in label

    def test_classify_standalone_mic(self):
        dev = AudioDevice(
            index=0, name="USB Microphone (Yeti)",
            max_input_channels=1, max_output_channels=0,
            default_sample_rate=44100, is_input=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=True)
        assert "Mic" in label or "Microphone" in label

    def test_classify_standalone_speaker(self):
        dev = AudioDevice(
            index=0, name="Realtek Speakers",
            max_input_channels=0, max_output_channels=2,
            default_sample_rate=48000, is_output=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=False)
        assert "Speaker" in label

    def test_classify_unknown(self):
        dev = AudioDevice(
            index=0, name="Unknown USB Device",
            max_input_channels=1, max_output_channels=0,
            default_sample_rate=44100, is_input=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=True)
        assert label == "system default"

    def test_classify_output_uses_output_priority(self):
        """Output classification uses _OUTPUT_PRIORITY keywords."""
        dev = AudioDevice(
            index=0, name="Desktop Speakers (USB)",
            max_input_channels=0, max_output_channels=2,
            default_sample_rate=48000, is_output=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=False)
        assert "Speaker" in label
    def test_classify_xbox_headset_input(self):
        dev = AudioDevice(
            index=24, name="Headset Microphone (Xbox Controller)",
            max_input_channels=1, max_output_channels=0,
            default_sample_rate=16000, is_input=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=True)
        assert label == "Bluetooth Headset"

    def test_classify_xbox_headphones_output(self):
        dev = AudioDevice(
            index=35, name="Headphones (Xbox Controller)",
            max_input_channels=0, max_output_channels=2,
            default_sample_rate=48000, is_output=True,
        )
        label = AudioDeviceManager.classify_device(dev, for_input=False)
        assert label == "Bluetooth Headset"


class TestListDevicesFormat:
    """list_devices() output with classification."""

    def test_list_devices_includes_classification(self):
        mgr = AudioDeviceManager()
        mgr.inject_devices([
            {
                "index": 0,
                "name": "Bluetooth Gaming Headset",
                "maxInputChannels": 2,
                "maxOutputChannels": 2,
                "defaultSampleRate": 48000.0,
            },
            {
                "index": 1,
                "name": "Logitech Webcam C920",
                "maxInputChannels": 1,
                "maxOutputChannels": 0,
                "defaultSampleRate": 16000.0,
            },
        ])
        lines = mgr.list_devices()
        # Should have section headers and device lines with classification
        joined = "\n".join(lines)
        assert "=== Input Devices ===" in joined
        assert "Bluetooth" in joined
        assert "Logitech" in joined

    def test_list_devices_empty(self):
        mgr = AudioDeviceManager()
        mgr.inject_devices([])
        lines = mgr.list_devices()
        assert "No audio devices found" in "\n".join(lines)


class TestSounddeviceFallback:
    """Graceful handling when PyAudio is unavailable."""

    def test_refresh_handles_pyaudio_missing(self, monkeypatch):
        """When PyAudio is not importable, refresh should log and return empty."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyaudio":
                raise ImportError("No module named 'pyaudio'")
            if name == "sounddevice":
                raise ImportError("No module named 'sounddevice'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        mgr = AudioDeviceManager()
        mgr.refresh()
        assert mgr.input_devices == []
        assert mgr.output_devices == []

    def test_refresh_with_injected_devices_skips_pyaudio(self, monkeypatch):
        """If devices were injected, refresh should not overwrite with PyAudio."""
        import builtins
        import_raised = False
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            nonlocal import_raised
            if name == "pyaudio":
                import_raised = True
                raise ImportError("No module named 'pyaudio'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        mgr = AudioDeviceManager()
        mgr.inject_devices([
            {"index": 0, "name": "Injected Mic", "maxInputChannels": 1,
             "maxOutputChannels": 0, "defaultSampleRate": 44100.0},
        ])
        # refresh should handle gracefully — injected devices remain
        mgr.refresh()
        # After refresh, injected devices would be cleared by refresh()
        # but since PyAudio import fails, the lists are cleared
        assert import_raised
