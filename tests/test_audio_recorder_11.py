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

Both sites now finish the failure the way the rest of the module does: log
it, and let the state the ledgers already hold answer for it.
"""

from __future__ import annotations

import threading

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
    _live_default_input(coreaudio_stub, 77)  # the default moved -> reset runs
    try:
        assert recorder.start_recording() is True  # the press survives
        worker = recorder.recording_thread
        assert worker is not None  # ... on the worker path, capture dropped
        assert _wait_until(lambda: recorder.recording is False, timeout=8.0)
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
