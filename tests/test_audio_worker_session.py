"""The worker's session head: a locked reset must fail the press, not open.

``refresh_recorder_session`` defers when another thread holds the reset lock
(``result.locked``): that thread may be mid-``terminate_all()``, and
``_dispose`` terminated the instance BEFORE ``forget``/``unbind`` ran, so the
pair the deferral snapshot carries can name an instance ``Pa_Terminate()``
was already called on while its index still resolves. The press that then
opens on that pair -- which is what the worker did -- records into a corpse
on healthy hardware. The deferral has to be a REPORT the worker acts on
(fail the press busy, no stream opened), not just a log line.
"""

from unittest.mock import MagicMock

from kuiskaus.audio_device_reset import SessionRefresh
from kuiskaus.audio_retry import MICROPHONE_BUSY
from kuiskaus.audio_worker_session import refresh_recorder_session
from tests.support_audio_recorder import (
    _cleanup_recorder,
    _make_pyaudio_instance,
    _make_recorder,
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
        thread = recorder.recording_thread
        assert thread is not None
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0), (
            "the worker opened anyway: the press never saw the deferral"
        )
        thread.join(timeout=5.0)
    finally:
        recorder._session._reset_lock.release()
        _cleanup_recorder(recorder)

    assert open_calls == [], "a locked deferral opened a stream on the torn pair"
    assert recorder.last_error == MICROPHONE_BUSY
    assert recorder.stream is None
    assert recorder.recording_thread is None
