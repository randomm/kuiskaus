"""Unit tests for kuiskaus.audio_retry.attempt_open_once (issue #40/43
lens review MEDIUM #1: the one-shot open attempt was extracted from
audio_recorder._attempt_open_once so audio_recorder.py stays under the
500-line module limit in AGENTS.md).

The function is a free function over the pyaudio module object and a
device-resolution callable, so tests drive it directly with mocks -- no
AudioRecorder construction, no threads.
"""

from unittest.mock import MagicMock


def _fake_pa_module():
    """A MagicMock standing in for the pyaudio module. The first
    PyAudio() call raises RuntimeError (simulating a Pa_Initialize
    failure); every later call returns the same pa_ok instance.
    Returns (module, pa_ok).
    """
    module = MagicMock(name="pyaudio")
    pa_ok = MagicMock(name="pa-ok")
    module.PyAudio = MagicMock(
        side_effect=[RuntimeError("Pa_Initialize failed"), pa_ok]
    )
    return module, pa_ok


def _find_device(pa):
    return 7


def test_attempt_open_once_success_returns_pa_stream_and_no_error():
    from kuiskaus.audio_retry import attempt_open_once

    pa_module, pa_ok = _fake_pa_module()
    pa_module.PyAudio = MagicMock(return_value=pa_ok)
    stream = MagicMock(name="stream")
    pa_ok.open.return_value = stream

    pa, got_stream, error = attempt_open_once(
        pa_module, 8, 1, 16000, 1024, _find_device, 4, 1, 0.0
    )

    assert error is None
    assert pa is pa_ok
    assert got_stream is stream
    pa_ok.open.assert_called_once_with(
        format=8,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=7,
        frames_per_buffer=1024,
    )


def test_attempt_open_once_construction_failure_returns_oserror_and_terminates_nothing():
    """A PyAudio() construction failure is wrapped in OSError (errno
    None) so the retry loop's RuntimeError check -- reserved for
    persistent device-lookup failures -- stays unambiguous. No PyAudio
    instance exists to terminate on this path."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = _fake_pa_module()[0]

    pa, got_stream, error = attempt_open_once(
        pa_module, 8, 1, 16000, 1024, _find_device, 4, 2, 0.0
    )

    assert pa is None
    assert got_stream is None
    assert isinstance(error, OSError)
    assert not isinstance(error, RuntimeError)
    assert not pa_module.PyAudio.return_value.terminate.called


def test_attempt_open_once_with_existing_pa_reuses_it_and_skips_resolution():
    """Issue #42: attempt 1 passes the cached PyAudio instance plus its
    cached device index. The cached instance must be reused verbatim --
    no PyAudio() construction, no re-resolution (find_default_input
    device never called) -- and the stream opened against the cached
    index."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    cached_pa = MagicMock(name="cached-pa")
    stream = MagicMock(name="stream")
    cached_pa.open.return_value = stream

    pa, got_stream, error = attempt_open_once(
        pa_module,
        8,
        1,
        16000,
        1024,
        _find_device,
        4,
        1,
        0.0,
        existing_pa=cached_pa,
        existing_device_index=42,
    )

    assert error is None
    assert pa is cached_pa
    assert got_stream is stream
    pa_module.PyAudio.assert_not_called()
    cached_pa.open.assert_called_once_with(
        format=8,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=42,
        frames_per_buffer=1024,
    )


def test_attempt_open_once_existing_pa_open_failure_does_not_terminate_cached():
    """Issue #42: a failed attempt-1 open on the CACHED instance must not
    terminate it -- the call site owns it and the retry loop (or the
    next recording) may still use it. The error is returned so the loop
    retries with a fresh instance."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    cached_pa = MagicMock(name="cached-pa")
    open_error = OSError("stale session")
    cached_pa.open.side_effect = open_error

    pa, got_stream, error = attempt_open_once(
        pa_module,
        8,
        1,
        16000,
        1024,
        _find_device,
        4,
        1,
        0.0,
        existing_pa=cached_pa,
        existing_device_index=42,
    )

    assert pa is None
    assert got_stream is None
    assert error is open_error
    cached_pa.terminate.assert_not_called()


def test_attempt_open_once_existing_pa_runtime_error_does_not_terminate_cached():
    """Issue #42: a device-lookup RuntimeError against the cached instance
    (no input device found -- persistent state) is returned for the
    call site's abort decision, and the cached instance is NOT
    terminated (the call site owns it)."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module = MagicMock(name="pyaudio")
    cached_pa = MagicMock(name="cached-pa")

    pa, got_stream, error = attempt_open_once(
        pa_module,
        8,
        1,
        16000,
        1024,
        lambda _pa: (_ for _ in ()).throw(RuntimeError("No input device found")),
        4,
        1,
        0.0,
        existing_pa=cached_pa,
    )

    assert pa is None
    assert got_stream is None
    assert isinstance(error, RuntimeError)
    cached_pa.terminate.assert_not_called()


def test_attempt_open_once_open_failure_terminates_the_failed_pa():
    """An open() OSError terminates the failed attempt's PyAudio (the
    attempt owns no stream) and returns the error so the loop retries."""
    from kuiskaus.audio_retry import attempt_open_once

    pa_module, pa_ok = _fake_pa_module()
    pa_module.PyAudio = MagicMock(return_value=pa_ok)
    open_error = OSError("device busy")
    pa_ok.open.side_effect = open_error

    pa, got_stream, error = attempt_open_once(
        pa_module, 8, 1, 16000, 1024, _find_device, 4, 2, 0.0
    )

    assert pa is None
    assert got_stream is None
    assert error is open_error
    pa_ok.terminate.assert_called_once()
