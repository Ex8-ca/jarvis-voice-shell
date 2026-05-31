"""Audio device enumeration and selection.

Priority chain (highest to lowest):
    1. Bluetooth gaming headset (keyword match)
    2. AirPods
    3. Standalone microphone / speakers
    4. Logitech webcam
    5. System default

Designed for Windows but uses PyAudio's cross-platform API.
Falls back to sounddevice if PyAudio is unavailable.
Supports dependency injection of raw device dicts for testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Priority-ordered keyword groups for auto-selection.
# Each group is a list of case-insensitive substrings — ALL must match.
_INPUT_PRIORITY: list[list[str]] = [
    ["xbox", "headset"],                              # Xbox wireless headset/controller path
    ["bluetooth", "headset"],                         # Bluetooth gaming headset
    ["headset", "microphone"],                        # Generic headset mic
    ["airpod"],                                       # AirPods (substring of "AirPods")
    ["microphone", "mic"],                            # Standalone mic
    ["logitech"],                                     # Logitech webcam/camera
]

_OUTPUT_PRIORITY: list[list[str]] = [
    ["xbox", "headphones"],                           # Xbox wireless headset/controller path
    ["bluetooth", "headset"],                         # Bluetooth gaming headset
    ["headphones"],                                   # Generic headset output
    ["airpod"],                                       # AirPods
    ["speaker"],                                      # Standalone speakers
    ["logitech"],                                     # Logitech webcam/camera
]

# Human-readable labels for each priority group.
_INPUT_LABELS: list[str] = [
    "Bluetooth Headset",
    "Bluetooth Headset",
    "Bluetooth Headset",
    "AirPods",
    "Standalone Mic",
    "Logitech Webcam",
]

_OUTPUT_LABELS: list[str] = [
    "Bluetooth Headset",
    "Bluetooth Headset",
    "Bluetooth Headset",
    "AirPods",
    "Standalone Speaker",
    "Logitech Webcam",
]

FALLBACK_LABEL = "system default"


@dataclass
class AudioDevice:
    """A single PyAudio device with parsed metadata."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    is_input: bool = False
    is_output: bool = False

    @property
    def display_name(self) -> str:
        return f"[{self.index}] {self.name}"

    def matches_keywords(self, keywords: list[str]) -> bool:
        """Check if device name contains ALL keywords (case-insensitive)."""
        name_lower = self.name.lower()
        return all(kw.lower() in name_lower for kw in keywords)


@dataclass
class AudioDeviceManager:
    """Enumerate and select audio devices by priority."""

    input_devices: list[AudioDevice] = field(default_factory=list)
    output_devices: list[AudioDevice] = field(default_factory=list)
    _pa: object = field(default=None, repr=False)  # PyAudio instance
    _injected: bool = field(default=False, repr=False)
    """True if devices were set via inject_devices (skip auto-refresh)."""

    # ------------------------------------------------------------------
    #  Injection — use raw device dicts (PyAudio key names) for tests
    # ------------------------------------------------------------------
    def inject_devices(self, raw_devices: list[dict]) -> None:
        """Populate input/output devices from a list of raw device dicts.

        Each dict should mirror PyAudio's get_device_info_by_index() output:
            index, name, maxInputChannels, maxOutputChannels, defaultSampleRate.

        This bypasses PyAudio entirely — useful for tests and environments
        where PyAudio is not installable.
        """
        self.input_devices.clear()
        self.output_devices.clear()
        self._injected = True

        for info in raw_devices:
            idx = int(info.get("index", 0))
            name = info.get("name", "")
            max_in = int(info.get("maxInputChannels", 0))
            max_out = int(info.get("maxOutputChannels", 0))
            rate = int(info.get("defaultSampleRate", 44100))

            if max_in > 0:
                self.input_devices.append(AudioDevice(
                    index=idx, name=name, max_input_channels=max_in,
                    max_output_channels=max_out, default_sample_rate=rate,
                    is_input=True,
                ))
            if max_out > 0:
                self.output_devices.append(AudioDevice(
                    index=idx, name=name, max_input_channels=max_in,
                    max_output_channels=max_out, default_sample_rate=rate,
                    is_output=True,
                ))

    # ------------------------------------------------------------------
    #  Refresh — real enumeration via PyAudio, sounddevice, or injected
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-enumerate audio devices from PyAudio, falling back to sounddevice.

        If neither library is available, logs a warning and leaves device lists
        empty (graceful degradation — the caller can check and handle).
        """
        self._injected = False

        # Try PyAudio first
        try:
            import pyaudio
            self._refresh_pyaudio(pyaudio)
            return
        except ImportError:
            logger.debug("PyAudio not available, trying sounddevice fallback...")

        # Fallback: sounddevice
        try:
            import sounddevice as sd
            self._refresh_sounddevice(sd)
            return
        except ImportError:
            logger.warning(
                "Neither PyAudio nor sounddevice is installed. "
                "Install one with: pip install pyaudio  OR  pip install sounddevice"
            )
            self.input_devices.clear()
            self.output_devices.clear()

    def _refresh_pyaudio(self, pyaudio) -> None:
        """Enumerate via PyAudio."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        self.input_devices.clear()
        self.output_devices.clear()

        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            name = info.get("name", "")
            max_in = info.get("maxInputChannels", 0)
            max_out = info.get("maxOutputChannels", 0)
            rate = int(info.get("defaultSampleRate", 44100))

            if max_in > 0:
                self.input_devices.append(AudioDevice(
                    index=i, name=name, max_input_channels=max_in,
                    max_output_channels=max_out, default_sample_rate=rate,
                    is_input=True,
                ))
            if max_out > 0:
                self.output_devices.append(AudioDevice(
                    index=i, name=name, max_input_channels=max_in,
                    max_output_channels=max_out, default_sample_rate=rate,
                    is_output=True,
                ))

    def _refresh_sounddevice(self, sd) -> None:
        """Enumerate via sounddevice (cross-platform fallback)."""
        self.input_devices.clear()
        self.output_devices.clear()

        try:
            devices = sd.query_devices()
        except Exception:
            logger.warning("sounddevice.query_devices() failed.", exc_info=True)
            return

        for idx, info in enumerate(devices):
            name = info.get("name", "")
            max_in = info.get("max_input_channels", 0)
            max_out = info.get("max_output_channels", 0)
            rate = int(info.get("default_samplerate", 44100))

            if max_in > 0:
                self.input_devices.append(AudioDevice(
                    index=idx, name=name, max_input_channels=max_in,
                    max_output_channels=max_out, default_sample_rate=rate,
                    is_input=True,
                ))
            if max_out > 0:
                self.output_devices.append(AudioDevice(
                    index=idx, name=name, max_input_channels=max_in,
                    max_output_channels=max_out, default_sample_rate=rate,
                    is_output=True,
                ))

    # ------------------------------------------------------------------
    #  Selection
    # ------------------------------------------------------------------
    def select_input(self, preferred_index: int | None = None) -> AudioDevice:
        """Select the best input device.

        Args:
            preferred_index: If set, use this exact device index (bypasses priority).

        Returns:
            The selected AudioDevice.

        Raises:
            RuntimeError: If no input devices are available.
        """
        if not self.input_devices and not self._injected:
            self.refresh()
        if not self.input_devices:
            raise RuntimeError("No input devices found. Check audio hardware.")

        if preferred_index is not None:
            for dev in self.input_devices:
                if dev.index == preferred_index:
                    logger.info("Using user-specified input device: %s", dev.display_name)
                    return dev
            logger.warning(
                "Preferred input index %d not found, falling back to auto-select.",
                preferred_index,
            )

        return self._prioritized_select(self.input_devices, _INPUT_PRIORITY)

    def select_output(self, preferred_index: int | None = None) -> AudioDevice:
        """Select the best output device.

        Args:
            preferred_index: If set, use this exact device index (bypasses priority).

        Returns:
            The selected AudioDevice.

        Raises:
            RuntimeError: If no output devices are available.
        """
        if not self.output_devices and not self._injected:
            self.refresh()
        if not self.output_devices:
            raise RuntimeError("No output devices found. Check audio hardware.")

        if preferred_index is not None:
            for dev in self.output_devices:
                if dev.index == preferred_index:
                    logger.info("Using user-specified output device: %s", dev.display_name)
                    return dev
            logger.warning(
                "Preferred output index %d not found, falling back to auto-select.",
                preferred_index,
            )

        return self._prioritized_select(self.output_devices, _OUTPUT_PRIORITY)

    @staticmethod
    def _prioritized_select(
        devices: list[AudioDevice], priority_groups: list[list[str]],
    ) -> AudioDevice:
        """Select first device matching a priority group, or fall back to first device."""
        for group in priority_groups:
            for dev in devices:
                if dev.matches_keywords(group):
                    logger.info("Auto-selected device (priority match): %s", dev.display_name)
                    return dev
        fallback = devices[0]
        logger.info("No priority match — using fallback device: %s", fallback.display_name)
        return fallback

    # ------------------------------------------------------------------
    #  Classification
    # ------------------------------------------------------------------
    @staticmethod
    def classify_device(device: AudioDevice, for_input: bool = True) -> str:
        """Return a human-readable priority label for a device.

        Args:
            device: The AudioDevice to classify.
            for_input: True to use input priority groups, False for output.

        Returns:
            Label like "Bluetooth Headset", "AirPods", "Standalone Mic", etc.,
            or "system default" if no priority group matches.
        """
        priority_groups = _INPUT_PRIORITY if for_input else _OUTPUT_PRIORITY
        labels = _INPUT_LABELS if for_input else _OUTPUT_LABELS

        for group, label in zip(priority_groups, labels):
            if device.matches_keywords(group):
                return label
        return FALLBACK_LABEL

    # ------------------------------------------------------------------
    #  Listing
    # ------------------------------------------------------------------
    def list_devices(self) -> list[str]:
        """Return formatted list of all input and output devices with classification."""
        if not self.input_devices and not self.output_devices and not self._injected:
            self.refresh()

        if not self.input_devices and not self.output_devices:
            return ["No audio devices found — install PyAudio or sounddevice."]

        lines = []
        lines.append("=== Input Devices ===")
        for dev in self.input_devices:
            label = self.classify_device(dev, for_input=True)
            lines.append(f"  {dev.display_name}  [{label}]")
        lines.append("=== Output Devices ===")
        for dev in self.output_devices:
            label = self.classify_device(dev, for_input=False)
            lines.append(f"  {dev.display_name}  [{label}]")
        return lines

    def list_devices_json(self) -> list[dict]:
        """Return device list as dicts for machine-readable output."""
        if not self.input_devices and not self.output_devices and not self._injected:
            self.refresh()
        result = []
        for dev in self.input_devices:
            result.append({
                "index": dev.index,
                "name": dev.name,
                "direction": "input",
                "max_channels": dev.max_input_channels,
                "sample_rate": dev.default_sample_rate,
                "classification": self.classify_device(dev, for_input=True),
            })
        for dev in self.output_devices:
            result.append({
                "index": dev.index,
                "name": dev.name,
                "direction": "output",
                "max_channels": dev.max_output_channels,
                "sample_rate": dev.default_sample_rate,
                "classification": self.classify_device(dev, for_input=False),
            })
        return result

    def __del__(self):
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
