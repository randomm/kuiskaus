"""A device that will not settle must not cost a rebuild on every press.

The reset's dedupe guard compares the live CoreAudio pair against the pair
the session was last rebuilt against. When CoreAudio is steady that is
exactly one rebuild per device move. It is NOT exactly one when the pair
itself oscillates -- an AirPod stream whose ``isAlive`` flag flips between
reads through a coreaudiod storm -- because each press then legitimately
sees a different pair, the guard cannot engage, and each press pays a full
terminate-all + ``Pa_Initialize`` + re-enumeration on the hotkey callback
thread. Nothing bounded that, so the "one rebuild" property issue #72 is
about was only ever true for a well-behaved CoreAudio.
"""

from unittest.mock import MagicMock

import kuiskaus.audio_device_reset as reset_module
from kuiskaus.audio_coreaudio import (
    kAudioDevicePropertyDeviceIsAlive,
    kAudioHardwarePropertyDefaultInputDevice,
)
from kuiskaus.audio_device import PyAudioRegistry
from tests.support_audio_recorder import _live_default_input


def _registry(pa=None):
    module = MagicMock(name="pyaudio-module")
    if pa is None:
        pa = MagicMock(name="PyAudioInstance")
    module.PyAudio = MagicMock(return_value=pa)
    return PyAudioRegistry(module), pa, module


def _find_device(_pa):
    return 3


def _oscillating_liveness(stub, device_id: int) -> None:
    """Report ``device_id`` as the default, with ``isAlive`` alternating on
    every read -- the storm shape that defeats the dedupe guard."""
    counter = {"n": 0}

    def _read(_object_id, address, _qs, _q, _s, _out):
        if address.mSelector == kAudioHardwarePropertyDefaultInputDevice:
            return 0, 4, device_id.to_bytes(4, "little")
        if address.mSelector == kAudioDevicePropertyDeviceIsAlive:
            counter["n"] += 1
            return 0, 4, int(counter["n"] % 2 == 0).to_bytes(4, "little")
        return 1852797079, 0, None  # 'who?' -- unknown property

    stub.AudioObjectGetPropertyData.side_effect = _read


def test_a_persistently_dead_default_rebuilds_once_not_once_per_press(
    coreaudio_stub,
):
    """The stable-pair baseline: the guard engages, and the presses after the
    first cost no teardown and no construction."""
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77, is_alive=False)
    registry.open(_find_device)

    results = [registry.refresh(_find_device) for _ in range(4)]

    assert all(r.busy is False for r in results)
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # only the open() above


def test_an_oscillating_default_is_bounded_rather_than_rebuilding_forever(
    coreaudio_stub,
):
    """with the pair changing on every read the guard cannot
    engage, so consecutive rebuilds have to be counted -- otherwise the
    session churns PortAudio for the life of the process and never says so."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    presses = reset_module.MAX_RESET_REBUILDS * 3

    for _ in range(presses):
        registry.refresh(_find_device)

    assert registry.needs_restart is True
    # Bounded: the rebuilds stop at the budget (plus the one open() above),
    # instead of one per press.
    assert module.PyAudio.call_count <= reset_module.MAX_RESET_REBUILDS + 1


def test_a_settled_device_is_recovered_by_the_retry_after_the_backoff(
    coreaudio_stub, monkeypatch
):
    """The bound defers, it does not give up: once the device stops
    oscillating and the backoff expires, the reset runs again and the
    restart verdict goes away."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_RESET_REBUILDS):
        registry.refresh(_find_device)
    assert registry.needs_restart is True

    _live_default_input(coreaudio_stub, 88, is_alive=True)  # it settled
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    fresh = MagicMock(name="PyAudioInstance-settled")
    module.PyAudio = MagicMock(return_value=fresh)
    result = registry.refresh(_find_device)

    assert result.pyaudio is fresh
    assert registry.needs_restart is False
    assert registry.device_id == 88
