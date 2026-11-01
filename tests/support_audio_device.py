"""Shared fixtures for the PyAudioRegistry tests.

The registry is the object under test in most of the ``test_audio_device*``
modules, and every one of them needs the same two things: a registry built
over a stubbed ``pyaudio`` module (so no PortAudio is initialised), and a
device resolver that always answers the same index. They lived as private
copies in each module until the copies started to drift.
"""

from unittest.mock import MagicMock

import kuiskaus.audio_device_reset as reset_module
from kuiskaus.audio_coreaudio import LiveDefaultInput
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


def _device_verdict_backoff_open(registry, monkeypatch):
    """Put a clean-teardown DEVICE verdict on the record with the backoff
    window held open, and the streak counting device 88 -- the shape the
    new-evidence exemption answers to.

    Every module that pins the exemption needs exactly this state (a verdict
    earned, a window that will not expire, a counted device the press is not
    looking at); it started as a private copy in one module and was about to
    be copied into the next.
    """
    monkeypatch.setattr(reset_module, "_now", lambda: 0.0)
    registry.rebuild.note_rebuild(88)
    registry.last_reset = LiveDefaultInput(88, True)
    registry.needs_restart = True
    registry.terminate_failures = 0
    registry.restart_retry_after = 1e12
