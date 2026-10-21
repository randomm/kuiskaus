"""Unit tests for kuiskaus.audio_device (issue #72).

No hardware and no CoreAudio framework: the autouse ``coreaudio_stub``
fixture owns the bridge, so a developer's real default input device can
never leak into these assertions.
"""

import threading
from unittest.mock import MagicMock

from kuiskaus.audio_device import (
    LiveDefaultInput,
    PyAudioRegistry,
    check_superseded,
    read_live_default_input,
    session_needs_reset,
)
from kuiskaus.audio_retry import registry_for
from tests.support_audio_recorder import _live_default_input

# 'duid' / 'lvly' as the stubbed bridge reports them (same ints the module
# uses -- a test that retypes them here would catch a constant typo).
DEFAULT_INPUT_SELECTOR = 1682533920
ALIVE_SELECTOR = 1818850926


def _registry(pa_module=None, pa=None):
    """A registry over a stubbed pyaudio module yielding ``pa`` (or one
    fresh mock) per construction."""
    module = pa_module or MagicMock(name="pyaudio-module")
    if pa is None:
        pa = MagicMock(name="PyAudioInstance")
    module.PyAudio = MagicMock(return_value=pa)
    return PyAudioRegistry(module), pa, module


def _find_device(_pa):
    return 3


# ---------------------------------------------------------------------------
# Live CoreAudio read
# ---------------------------------------------------------------------------


def test_no_evidence_from_coreaudio_reads_as_none(coreaudio_stub):
    # The stub's default is "every read fails" -- a nonzero OSStatus is
    # CoreAudio's way of saying "no answer", which must not become
    # "the device did not change".
    assert read_live_default_input() is None


def test_live_read_returns_default_id_and_liveness(coreaudio_stub):
    _live_default_input(coreaudio_stub, 96)
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=True)


def test_live_read_reports_dead_device(coreaudio_stub):
    _live_default_input(coreaudio_stub, 96, is_alive=False)
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=False)


def test_failed_liveness_read_doubts_the_device(coreaudio_stub):
    """A device whose liveness cannot be proven is treated as NOT alive:
    the safe response to an unproven device is a rebuild, not a stream."""
    from kuiskaus import audio_device

    def _read(object_id, address, _qs, _q, _s, _out):
        if address.mSelector == DEFAULT_INPUT_SELECTOR:
            return 0, 4, (96).to_bytes(4, audio_device.sys.byteorder)
        return 1852797029, 0, None  # '!sta' -- read failed

    coreaudio_stub.AudioObjectGetPropertyData.side_effect = _read
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=False)


def test_non_bytes_property_data_is_no_evidence(coreaudio_stub):
    """A successful status with a non-bytes out-buffer is not a value."""
    from kuiskaus import audio_device

    coreaudio_stub.AudioObjectGetPropertyData.return_value = (0, 4, None)
    assert audio_device._read_uint32(1, DEFAULT_INPUT_SELECTOR, 0, 0) is None


# ---------------------------------------------------------------------------
# The change predicate
# ---------------------------------------------------------------------------


def test_no_evidence_never_resets():
    assert session_needs_reset(None, 42) is False
    assert session_needs_reset(None, None) is False


def test_unbound_session_binds_on_first_evidence():
    assert session_needs_reset(LiveDefaultInput(7, True), None) is True


def test_same_alive_device_needs_no_reset():
    assert session_needs_reset(LiveDefaultInput(7, True), 7) is False


def test_moved_default_or_dead_device_resets():
    assert session_needs_reset(LiveDefaultInput(8, True), 7) is True
    assert session_needs_reset(LiveDefaultInput(7, False), 7) is True


# ---------------------------------------------------------------------------
# The registry: every instance is reachable by terminate_all()
# ---------------------------------------------------------------------------


def test_terminate_all_terminates_every_instance_and_clears():
    registry, first, module = _registry()
    second = MagicMock(name="PyAudioInstance-2")
    registry.register(first)
    registry.register(second)
    registry.register(second)  # idempotent -- no double-terminate

    registry.terminate_all()

    first.terminate.assert_called_once()
    second.terminate.assert_called_once()
    registry.terminate_all()  # nothing left to terminate
    second.terminate.assert_called_once()
    assert module.PyAudio.call_count == 0  # terminate_all never constructs


def test_terminate_all_survives_a_failing_instance():
    """One broken terminate() must not strand the instances after it --
    a single live instance keeps PortAudio's init count above zero."""
    registry, bad, _ = _registry()
    good = MagicMock(name="PyAudioInstance-good")
    bad.terminate.side_effect = OSError("PortAudio still busy")
    registry.register(bad)
    registry.register(good)

    registry.terminate_all()

    good.terminate.assert_called_once()


def test_terminate_unregisters_so_the_instance_is_not_terminated_twice():
    registry, pa, _ = _registry()
    registry.register(pa)
    registry.terminate(pa)
    pa.terminate.assert_called_once()
    registry.terminate_all()
    pa.terminate.assert_called_once()


def test_register_stamps_the_owning_registry():
    """The worker's reset is handed an instance, not the session, so the
    instance must be able to find its registry (issue #72)."""
    registry, pa, _ = _registry()
    registry.register(pa)
    assert registry_for(pa) is registry
    assert registry_for(MagicMock(name="unowned")) is None
    assert registry_for(None) is None


def test_open_binds_the_instance_and_the_live_device(coreaudio_stub):
    _live_default_input(coreaudio_stub, 55)
    registry, pa, module = _registry()

    assert registry.open(_find_device) is True
    assert registry.pyaudio is pa
    assert registry.device_id == 55
    assert registry.input_device_index == 3
    assert module.PyAudio.call_count == 1
    assert registry_for(pa) is registry


def test_open_failure_leaves_session_unbound():
    module = MagicMock(name="pyaudio-module")
    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))
    registry = PyAudioRegistry(module)

    assert registry.open(_find_device) is False
    assert registry.pyaudio is None
    assert registry.input_device_index is None
    assert registry.device_id is None


# ---------------------------------------------------------------------------
# The reset
# ---------------------------------------------------------------------------


def test_reset_is_skipped_without_live_evidence(coreaudio_stub):
    """The no-reset fast path: a transient CoreAudio failure must not cost
    a session rebuild on every press."""
    registry, pa, module = _registry()
    registry.open(_find_device)

    result = registry.refresh(_find_device)

    assert result.busy is False
    assert result.pyaudio is pa
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # only the original open()


def test_moved_default_rebuilds_exactly_one_instance(coreaudio_stub):
    registry, stale, module = _registry()
    registry.open(_find_device)  # baseline: whatever the stub said then
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]  # the reset's one construction
    _live_default_input(coreaudio_stub, 77)

    result = registry.refresh(_find_device)

    stale.terminate.assert_called_once()
    assert result.pyaudio is fresh
    assert result.input_device_index == 3
    assert registry.device_id == 77
    # The whole point of issue #72: one rebuild, not one per attempt.
    assert module.PyAudio.call_count == 2


def test_reset_when_engine_never_comes_back_leaves_session_unbound(coreaudio_stub):
    registry, _, module = _registry()
    registry.open(_find_device)
    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))
    _live_default_input(coreaudio_stub, 77)

    result = registry.refresh(_find_device)

    assert result.pyaudio is None
    assert result.input_device_index is None
    assert result.busy is False
    assert registry.device_id is None


def test_warm_stream_that_will_not_stop_fails_the_press(coreaudio_stub):
    """A warm capture that would not stop means PortAudio is still up, so
    the session is left whole rather than half-reset -- the next press
    asks again instead of streaming into a stale device."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    warm = MagicMock(name="warm")
    warm.stop.return_value = False

    result = registry.refresh(_find_device, warm)

    assert result.busy is True
    assert result.pyaudio is pa
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1


def test_warm_capture_is_stopped_then_restarted(coreaudio_stub):
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]
    warm = MagicMock(name="warm")
    warm.stop.return_value = True
    restart = MagicMock(name="restart-warm")

    result = registry.refresh(_find_device, warm, restart)

    warm.stop.assert_called_once()  # BEFORE terminating: closing the last
    # stream is what releases PortAudio's per-stream init reference
    pa.terminate.assert_called_once()
    restart.assert_called_once()
    assert result.busy is False
    assert result.pyaudio is not pa


# ---------------------------------------------------------------------------
# Adoption and the supersede check
# ---------------------------------------------------------------------------


def _adopt(registry, new_pa, stream, superseded, stream_open, capture_rate=16000):
    published = {}
    ok = registry.adopt(
        threading.Lock(),
        lambda: superseded[0],
        lambda rate: published.__setitem__("rate", rate),
        lambda: stream_open[0],
        new_pa,
        stream,
        capture_rate,
        2,
        0.0,
        4,
    )
    return ok, published


def test_adopt_publishes_instance_and_rate():
    registry, live, _ = _registry()
    registry.open(_find_device)
    stream = MagicMock(name="stream")

    ok, published = _adopt(registry, live, stream, [False], [False])

    assert ok is True
    assert registry.pyaudio is live
    assert published["rate"] == 16000
    # The retry reused the live instance, so nothing may be terminated --
    # the stream is open on it (issue #72's invariant).
    live.terminate.assert_not_called()
    stream.close.assert_not_called()


def test_adopt_disposes_the_instance_a_reset_displaced():
    registry, stale, module = _registry()
    registry.open(_find_device)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio = MagicMock(return_value=fresh)
    registry.register(fresh)

    ok, _ = _adopt(registry, fresh, MagicMock(name="stream"), [False], [False])

    assert ok is True
    stale.terminate.assert_called_once()


def test_superseded_adopt_closes_stream_and_spares_the_live_session():
    registry, live, _ = _registry()
    registry.open(_find_device)
    stream = MagicMock(name="stream")

    ok, _ = _adopt(registry, live, stream, [True], [False])

    assert ok is False
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    live.terminate.assert_not_called()  # a later press still needs it


def test_superseded_adopt_disposes_an_untracked_fresh_instance():
    registry, _, _ = _registry()
    registry.open(_find_device)
    orphan = MagicMock(name="PyAudioInstance-orphan")

    ok, _ = _adopt(registry, orphan, MagicMock(name="stream"), [True], [False])

    assert ok is False
    orphan.terminate.assert_called_once()


def test_check_superseded_reports_and_logs(capsys):
    assert check_superseded(threading.Lock(), lambda: True, 2, 4, 0.0) is True
    assert check_superseded(threading.Lock(), lambda: False, 2, 4, 0.0) is False
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[audio.retry]")
    ]
    assert len(lines) == 1
    assert "action=abort" in lines[0]
