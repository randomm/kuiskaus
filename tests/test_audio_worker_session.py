"""The openers' session head: a locked reset must fail them, not open.

``refresh_recorder_session`` defers when another thread holds the reset lock
(``result.locked``): that thread may be mid-``terminate_all()``, and
``_dispose`` terminated the instance BEFORE ``forget``/``unbind`` ran, so the
pair the deferral snapshot carries can name an instance ``Pa_Terminate()``
was already called on while its index still resolves. The press that then
opens on that pair -- which is what the worker did -- records into a corpse
on healthy hardware. The deferral has to be a REPORT the caller acts on
(fail the press busy, no stream opened), not just a log line.

Both openers are covered here because both enter through the same call: the
per-press worker (``prepare_worker_session``) and the keep-warm capture's own
thread (``open_warm_capture_stream``, which the capture re-enters on every
reopen). The warm path read the bool and threw it away, so a warm open landed
on the torn pair exactly as the worker's used to.
"""

import threading
from unittest.mock import MagicMock

import pytest

from kuiskaus import audio_retry, audio_warm_arm
from kuiskaus.audio_device_reset import SessionRefresh
from kuiskaus.audio_retry import MICROPHONE_BUSY
from kuiskaus.audio_worker_session import refresh_recorder_session
from tests.support_audio_recorder import (
    _cleanup_recorder,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
)


def test_refresh_reports_a_locked_reset_as_a_deferral(audio_recorder_module):
    """The return value is the contract: True = the reset is somebody
    else's right now and the caller must not treat the snapshot it carries
    as usable. A ledger busy (False here) proceeds on its pair -- that is
    what the warm path and the press have always done."""
    recorder = _make_recorder(
        audio_recorder_module, init_probe=_make_pyaudio_instance(0)
    )
    registry = recorder._session
    registry._reset_lock.acquire()
    try:
        assert refresh_recorder_session(recorder) is True
    finally:
        registry._reset_lock.release()

    # A ledger deferral is not a locked one: it writes its pair back and
    # the caller proceeds.
    deferred = MagicMock(name="PyAudioInstance-deferred")
    recorder._session.refresh = lambda *_a, **_k: SessionRefresh(deferred, 2, busy=True)
    assert refresh_recorder_session(recorder) is False


def test_a_locked_reset_fails_the_worker_press_busy_without_opening(
    audio_recorder_module, monkeypatch
):
    """With ``_reset_lock`` held by another thread (a rebuild may be mid
    ``terminate_all``), a worker press must NOT reach ``attempt_open_once``
    -- opening now can re-adopt the (corpse, live index) pair the deferral
    snapshot carries. The press comes back failed as busy through the same
    generation-gated path the exhausted retry loop uses."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(module, init_probe=pa)

    open_calls: list = []
    monkeypatch.setattr(
        module,
        "attempt_open_once",
        lambda *_args, **_kwargs: (
            open_calls.append(1)
            or (None, None, OSError("must not open while the reset is locked"), None)
        ),
    )

    recorder._session._reset_lock.acquire()  # the holder is mid-rebuild
    try:
        assert recorder.start_recording() is True
        # Wait for the STATE, not for the worker's name: this press fails in
        # microseconds (the deferral is a non-blocking refusal), so on a busy
        # machine the worker can clear ``recording``/``recording_thread``
        # before this line reads them. Asserting the name was still set made
        # the test fail exactly when the guard was fastest.
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0), (
            "the worker opened anyway: the press never saw the deferral"
        )
        thread = recorder.recording_thread
        if thread is not None:
            thread.join(timeout=5.0)
    finally:
        recorder._session._reset_lock.release()
        _cleanup_recorder(recorder)

    assert open_calls == [], "a locked deferral opened a stream on the torn pair"
    assert recorder.last_error == MICROPHONE_BUSY
    assert recorder.stream is None
    assert recorder.recording_thread is None


def test_a_locked_reset_never_opens_the_keep_warm_stream_on_the_torn_pair(
    audio_recorder_module, monkeypatch
):
    """The warm path had the same hazard and no guard:
    ``open_warm_capture_stream`` called the reset and dropped its answer,
    then opened on the snapshot anyway. With ``_reset_lock`` held by another
    thread (a rebuild may be mid-``terminate_all``) the warm open must not be
    attempted at all -- neither ``open_warm_stream`` nor the
    ``attempt_open_once`` beneath it -- and the failure has to come back as
    the one no-stream result ``WarmCapture._run`` already handles, so it
    waits out its reopen delay instead of reading a stream nobody opened.
    """
    recorder = _make_recorder(
        audio_recorder_module, init_probe=_make_pyaudio_instance(0, rate=16000.0)
    )
    attempts: list = []
    monkeypatch.setattr(
        audio_warm_arm,
        "open_warm_stream",
        lambda *_a, **_k: attempts.append("open_warm_stream") or (None, None, 0),
    )
    monkeypatch.setattr(
        audio_retry,
        "attempt_open_once",
        lambda *_a, **_k: (
            attempts.append("attempt_open_once")
            or (None, None, OSError("must not open while the reset is locked"), None)
        ),
    )

    recorder._session._reset_lock.acquire()  # the holder is mid-rebuild
    try:
        with pytest.raises(RuntimeError):
            recorder._open_warm_stream()
    finally:
        recorder._session._reset_lock.release()
        _cleanup_recorder(recorder)

    assert attempts == [], f"a locked deferral warmed open on the torn pair: {attempts}"


def test_the_keep_warm_loop_retries_after_a_locked_reset_deferral(
    audio_recorder_module, coreaudio_stub, monkeypatch
):
    """The deferral is a wait, not a give-up. The capture's thread asks for
    the session, is refused, and comes back after its reopen delay: once the
    holder is gone the same capture opens and delivers, so a keep-warm press
    that lands during someone else's teardown is late by one reopen instead
    of staying silently OFF behind a checked menu item for the session.
    """
    release = threading.Event()
    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.return_value = _streaming_stream(release)
    recorder = _make_recorder(audio_recorder_module, init_probe=pa)

    asked: list = []
    real_refresh = audio_warm_arm.refresh_recorder_session

    def _spy(recorder_arg, blocked=None):
        asked.append(1)
        return real_refresh(recorder_arg, blocked)

    monkeypatch.setattr(audio_warm_arm, "refresh_recorder_session", _spy)

    recorder._session._reset_lock.acquire()  # the holder is mid-rebuild
    try:
        recorder.set_keep_warm(True)
        warm = recorder._warm
        assert warm is not None
        assert _wait_until(lambda: len(asked) >= 1, timeout=5.0), (
            "the warm open never asked for the session"
        )
        pa.open.assert_not_called()
        assert warm.ready is False
    finally:
        recorder._session._reset_lock.release()

    try:
        # The holder is gone; the loop's reopen delay (2 s) is the only thing
        # left standing between the capture and its stream.
        assert _wait_until(lambda: warm.ready, timeout=8.0), (
            "the deferred warm capture never retried"
        )
        assert len(asked) >= 2, "the warm loop did not ask again after the deferral"
        assert pa.open.call_count >= 1
    finally:
        release.set()
        _cleanup_recorder(recorder)
