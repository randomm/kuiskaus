"""Shared fixtures for the PyAudioRegistry tests.

The registry is the object under test in most of the ``test_audio_device*``
modules, and every one of them needs the same two things: a registry built
over a stubbed ``pyaudio`` module (so no PortAudio is initialised), and a
device resolver that always answers the same index. They lived as private
copies in each module until the copies started to drift.
"""

from unittest.mock import MagicMock

from kuiskaus.audio_device import PyAudioRegistry


def _registry(pa_module=None, pa=None):
    """A registry over a stubbed pyaudio module yielding ``pa`` (or one
    fresh mock) per construction."""
    module = pa_module or MagicMock(name="pyaudio-module")
    if pa is None:
        pa = MagicMock(name="PyAudioInstance")
    module.PyAudio = MagicMock(return_value=pa)
    return PyAudioRegistry(module), pa, module


def _find_device(_pa):
    """A resolver that always answers index 3, for tests that only need the
    call to succeed."""
    return 3
