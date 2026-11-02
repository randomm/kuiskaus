"""The press must survive a re-arm that fails, and a worker that raises late.

Two places where a failure inside the recovery path became a failure of the
press:

* ``reset_session`` ends by handing the keep-warm capture back with
  ``rearm_warm(True)``, and ``WarmArm`` re-raises a capture whose
  ``start()`` failed -- an arming window the ledger already treats as a
  blocker, so the undo is already done and the recovery ask already filed.
  Re-raising through the reset put that on the hotkey callback thread, which
  runs the press: the user's press died with an traceback instead of
  recording on the worker path.
* ``_recording_worker`` guards its head but not the rest of its body, so a
  raise after the stream opened -- in the adoption, in the queue -- left
  ``recording`` True with the thread gone.
* The worker head ran the essential session reset and the auxiliary
  keep-warm ``recover()`` re-arm in ONE guard, so a failed re-arm (its
  ``capture.start()`` raising) failed the user's press through
  ``fail_worker_session`` -- the press died over the capture it was not
  using (issue #72).

All three sites now finish the failure the way the rest of the module does: log
it, and let the state the ledgers already hold answer for it.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from tests.support_audio_recorder import (
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
    _warm_recorder,
)


def _find_device(_pa):
    return 3


def test_a_failed_rearm_does_not_kill_the_press(audio_recorder_module, coreaudio_stub):
    """The re-arm is the last act of the reset and it can fail -- thread
    creation does fail -- and the press the user made has to go through
    regardless: the recovery flag brings the capture back on a later press."""
    module = audio_recorder_module
    release = threading.Event()
    recorder, _pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)

    # From here on every capture the recorder builds fails to start.
    class _UnstartableCapture:
        def __init__(self, _open_stream, _chunk_size):
            self.capture_rate = None
            self.started = False
            self._stop = threading.Event()
            self.thread = None

        def start(self):
            raise RuntimeError("can't start new thread")

        def stop(self, timeout: float = 2.0) -> bool:
            self._stop.set()
            return True

        def begin(self, *_args, **_kwargs):
            pass

        def end(self):
            pass

    module.WarmCapture = _UnstartableCapture
    # The reset constructs a fresh instance for the rebuilt session; give
    # its open the streaming stream, so the surviving press records bytes
    # rather than reading an unconstrained mock.
    reset_pa = _make_pyaudio_instance(0, rate=16000.0)
    reset_pa.open.return_value = _streaming_stream(release)
    module.pyaudio.PyAudio = MagicMock(return_value=reset_pa)
    _live_default_input(coreaudio_stub, 77)  # the default moved -> reset runs
    try:
        assert recorder.start_recording() is True  # the press survives
        # The worker's NAME is not read here: right after an admitted press
        # that read races a worker finishing and clearing recording_thread
        # in microseconds (the class 730c65f removed from recorder_7/8).
        # What pins the press to the worker path is the pair only that path
        # produces: it rode no capture, and it opened and ADOPTED its own
        # stream below -- a warm press routes the capture's ring and never
        # sets recorder.stream, so the adoption answers "a worker ran".
        assert recorder._press_warm is None, "the press rode the keep-warm route"
        # Survives means records: the worker adopted its stream and the
        # press is not blamed for the re-arm it was never using.
        assert _wait_until(lambda: recorder.stream is not None, timeout=8.0), (
            "the press never adopted a stream: no worker opened for it"
        )
        assert recorder.recording is True
        assert recorder.last_error is None
        # The failed arming asked for recovery, so keep-warm is not left off
        # behind a checked menu for the life of the process.
        assert recorder._session.warm._recover_pending is True
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_a_raise_after_the_stream_opened_fails_the_press(
    audio_recorder_module, monkeypatch
):
    """The worker's guard has to cover the whole body, not just its head: a
    raise past the guarded section would leave ``recording`` True with no
    thread left to clear it."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(module, init_probe=pa)

    real = recorder._open_stream_with_retry

    def _open_then_raise(my_gen: int):
        stream = real(my_gen)
        if stream is not None:
            raise OSError("the queue rejected the chunk")
        return stream

    monkeypatch.setattr(recorder, "_open_stream_with_retry", _open_then_raise)
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0), (
            "the worker died past its guard: recording stayed True and "
            "every later press is refused as stale"
        )
        assert recorder.recording_thread is None
        assert recorder.last_error is not None
    finally:
        _cleanup_recorder(recorder)


def test_a_failed_keep_warm_recovery_does_not_fail_the_press(
    audio_recorder_module, coreaudio_stub
):
    """The worker head ran the session reset and the auxiliary keep-warm
    re-arm in one guard, so a failed re-arm -- WarmArm._arm_locked re-raises
    a capture whose start() failed -- failed the user's press through
    fail_worker_session. The press does not use the keep-warm capture at all
    (it opened its own stream on the worker path); it must survive the
    failure of the thing it is not using, with the recovery flag left for a
    later press to pick up."""
    module = audio_recorder_module
    release = threading.Event()
    recorder, _pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)

    # The stuck-read already went back, and the last failed arming filed the
    # recovery ask; every capture the recorder builds from here on fails to
    # start, as it did under the thread storm.
    dead = recorder._warm_arm.take()
    assert recorder._session.warm.drop(dead) is True
    recorder._session.warm.need_recovery()

    class _UnstartableCapture:
        def __init__(self, _open_stream, _chunk_size):
            self.capture_rate = None
            self.started = False
            self._stop = threading.Event()
            self.thread = None

        def start(self):
            raise RuntimeError("can't start new thread")

        def stop(self, timeout: float = 2.0) -> bool:
            self._stop.set()
            return True

        def begin(self, *_args, **_kwargs):
            pass

        def end(self):
            pass

    module.WarmCapture = _UnstartableCapture
    try:
        assert recorder.start_recording() is True  # the press survives
        # The press is recording: the worker adopted its stream, and the
        # failed re-arm left no trace on the press's own error surface.
        assert _wait_until(lambda: recorder.stream is not None, timeout=8.0), (
            "the press died in the worker head: the auxiliary keep-warm "
            "re-arm took the user's recording down with it"
        )
        assert recorder.recording is True
        assert recorder.last_error is None
        # The failed arming still asked for recovery, so a later press
        # brings keep-warm back behind the checked menu item.
        assert recorder._session.warm._recover_pending is True
    finally:
        release.set()
        _cleanup_recorder(recorder)
