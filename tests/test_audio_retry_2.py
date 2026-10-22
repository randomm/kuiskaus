"""test_audio_retry tests, part 2 (issue #72).

Issue #72's registry contract for ``attempt_open_once``: the retry loop
constructs no new PyAudio while a live instance exists, every instance
the attempt itself constructs is registered (and disposed THROUGH the
registry, so terminate_all() can never double-terminate or miss one),
and a device index resolved on an unbound session is handed back via
``bind_index``. Same free-function harness as part 1: mocks only, no
AudioRecorder construction, no threads.
"""

from unittest.mock import MagicMock


def _find_device(pa):
    return 7


def _registry():
    """A stand-in session registry exposing only the surface
    attempt_open_once touches (register / terminate / bind_index)."""
    registry = MagicMock(name="registry")
    registry._instances = []
    return registry


def test_attempt_open_once_reuses_existing_pa_on_every_attempt_and_constructs_none():
    """Issue #72 AC "_open_stream_with_retry reuses the single live PyAudio
    instance on all attempts": drive attempt_open_once the way the retry
    loop does -- existing_pa plus its index on attempts 1..N -- and no
    PyAudio is ever constructed while the live one is alive. The pre-#72
    loop constructed a fresh instance on attempts 2..N, which kept
    PortAudio's init count above zero and replayed the frozen device
    list on every press."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    live_pa = MagicMock(name="live-pa")
    live_pa.get_device_info_by_index.return_value = {
        "index": 42,
        "defaultSampleRate": 48000.0,
    }
    stream = MagicMock(name="stream")
    # Attempts 1 and 2 hit the storm; attempt 3 opens.
    live_pa.open.side_effect = [OSError("-9986"), OSError("-9986"), stream]

    for attempt in (1, 2):
        pa, got_stream, error, rate = attempt_open_once(
            pa_module,
            8,
            1,
            16000,
            1024,
            _find_device,
            4,
            attempt,
            0.0,
            existing_pa=live_pa,
            existing_device_index=42,
        )
        assert pa is None
        assert got_stream is None
        assert isinstance(error, OSError)
        assert rate is None
        live_pa.terminate.assert_not_called()  # the call site owns the live one

    pa, got_stream, error, rate = attempt_open_once(
        pa_module,
        8,
        1,
        16000,
        1024,
        _find_device,
        4,
        3,
        0.0,
        existing_pa=live_pa,
        existing_device_index=42,
    )

    assert error is None
    assert pa is live_pa
    assert got_stream is stream
    assert rate == 48000
    # Three attempts, zero constructions: the loop re-enumerates only via
    # the device reset, never via a retry.
    pa_module.PyAudio.assert_not_called()
    assert live_pa.open.call_count == 3


def test_attempt_open_once_registers_a_constructed_instance():
    """Issue #72: an instance the attempt itself constructs must land in
    the session registry immediately -- an untracked live instance is
    exactly what terminate_all() can never reach, which is the whole
    frozen-snapshot bug."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    pa_ok = MagicMock(name="pa-ok")
    pa_ok.get_device_info_by_index.return_value = {"index": 7}
    pa_ok.open.return_value = MagicMock(name="stream")
    pa_module.PyAudio = MagicMock(return_value=pa_ok)
    registry = _registry()

    pa, _stream, error, _rate = attempt_open_once(
        pa_module, 8, 1, 16000, 1024, _find_device, 4, 1, 0.0, registry=registry
    )

    assert error is None
    assert pa is pa_ok
    registry.register.assert_called_once_with(pa_ok)


def test_attempt_open_once_open_failure_disposes_the_instance_via_the_registry():
    """Issue #72: a failed constructed instance is terminated THROUGH the
    registry (registry.terminate), which untracks it on success -- a
    terminated instance left registered would be terminated a second
    time by terminate_all()."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    pa_ok = MagicMock(name="pa-ok")
    pa_ok.get_device_info_by_index.return_value = {"index": 7}
    pa_ok.open.side_effect = OSError("device busy")
    pa_module.PyAudio = MagicMock(return_value=pa_ok)
    registry = _registry()

    pa, _stream, error, _rate = attempt_open_once(
        pa_module, 8, 1, 16000, 1024, _find_device, 4, 2, 0.0, registry=registry
    )

    assert pa is None
    assert isinstance(error, OSError)
    # Disposal goes THROUGH the registry (which untracks on success) --
    # a bare terminate_quietly would leave a terminated instance in the
    # list for terminate_all() to terminate a second time.
    registry.terminate.assert_called_once_with(pa_ok)
    registry.register.assert_called_once_with(pa_ok)


def test_attempt_open_once_binds_the_resolved_index_to_the_registry():
    """Issue #72: when the session was unbound, the index this attempt
    resolved must be handed back via bind_index -- otherwise it dies with
    the attempt and every later press pays a fresh enumeration. It goes
    back PAIRED with the instance it was resolved against (issue #72
    review, round 1): a PortAudio index is a fact about one instance's
    frozen enumeration, and an unpaired binding is what pinned a session
    to a disconnected device forever."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    pa_ok = MagicMock(name="pa-ok")
    pa_ok.get_device_info_by_index.return_value = {"index": 7}
    pa_ok.open.return_value = MagicMock(name="stream")
    pa_module.PyAudio = MagicMock(return_value=pa_ok)
    registry = _registry()

    _pa, _stream, error, _rate = attempt_open_once(
        pa_module, 8, 1, 16000, 1024, _find_device, 4, 1, 0.0, registry=registry
    )

    assert error is None
    registry.bind_index.assert_called_once_with(7, pa_ok)


def test_attempt_open_once_reuse_path_never_touches_the_registry():
    """Issue #72: reuse is not a construction -- with existing_pa plus a
    known index the attempt must not register, terminate, or bind
    anything on the session."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    live_pa = MagicMock(name="live-pa")
    live_pa.get_device_info_by_index.return_value = {"index": 42}
    live_pa.open.return_value = MagicMock(name="stream")
    registry = _registry()

    pa, _stream, error, _rate = attempt_open_once(
        pa_module,
        8,
        1,
        16000,
        1024,
        _find_device,
        4,
        1,
        0.0,
        existing_pa=live_pa,
        existing_device_index=42,
        registry=registry,
    )

    assert error is None
    assert pa is live_pa
    registry.register.assert_not_called()
    registry.terminate.assert_not_called()
    registry.bind_index.assert_not_called()
