"""The press that pays for a warm-path rebuild keeps its take (issue #72).

With keep-warm on, the press itself runs the live-default reset, and
``_rebuild`` ended by arming a fresh ``WarmCapture`` for the session it had
just rebuilt. ``start_recording`` then re-read ``self._warm`` -- the NEW
capture -- and routed the press through ``warm.begin(sink, ...)``. But the
new capture had not opened its stream yet (its run loop sets ``_ready`` and
seeds the ring only after ``_open_stream()`` returns), so ``begin()`` seeded
the press from an EMPTY ring: the take came back empty and
``stop_recording`` called it "microphone busy — recording did not start"
for a reset that had SUCCEEDED. The worker path has no such hole -- it
opens and reads on itself.

The press that moves the default now records on the worker, and keep-warm
returns through the worker's own ``WarmKeep.recover()``. These drive the
public press path and assert the take, which no earlier test did (the only
content assertions were on the stuck-open paths).
"""

import threading

from tests.support_audio_recorder import (
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
)


def test_the_press_that_triggers_a_warm_rebuild_delivers_its_take(
    audio_recorder_module, coreaudio_stub
):
    """Press on a keep-warm recorder while the default moves: the rebuild
    runs, the press records (non-empty take, no "busy" lie), and the
    session really was replaced."""
    module = audio_recorder_module
    release = threading.Event()
    stale = _make_pyaudio_instance(0, rate=16000.0)
    stale.open.return_value = _streaming_stream(release)
    rebuilt = _make_pyaudio_instance(5, rate=16000.0)
    rebuilt.open.return_value = _streaming_stream(release)
    recorder = _make_recorder(module, init_probe=stale, extra_pa=rebuilt)
    recorder.set_keep_warm(True)
    assert _wait_until(lambda: recorder._warm.ready)

    _live_default_input(coreaudio_stub, 77)  # the default moved under us

    try:
        assert recorder.start_recording() is True
        # The worker's NAME is not read here: right after an admitted press
        # it races a worker finishing and clearing recording_thread (the
        # class 730c65f removed). The routing the name read was standing in
        # for is pinned by the state instead: the press rode NO capture,
        # and the chunks below can only come from the stream ITS read loop
        # put on the queue -- a press routed into the capture armed
        # mid-reset would have been seeded from that capture's empty ring
        # and returned an empty take (the bug this test exists for).
        assert recorder._press_warm is None, (
            "the press was routed into the capture armed mid-reset: it had "
            "not opened its stream, so begin() seeded the take from an "
            "empty ring"
        )
        assert _wait_until(lambda: not recorder.audio_queue.empty(), timeout=8.0)
        audio = recorder.stop_recording()
        assert audio.size > 0, "the take that paid for the rebuild came back empty"
        assert recorder.last_error is None
        # The worker exited -- answered by the ledger (the registry object
        # survives the rebuild; only its instance/index are swapped), which
        # asks the same question `not worker.is_alive()` used to, without
        # reading the name a completing worker may have cleared.
        assert _wait_until(
            lambda: not recorder._session.workers.any_may_be_native(), timeout=8.0
        ), "the worker this press started never exited"
        # The rebuild itself: one teardown, one construction, the session
        # now names the fresh instance.
        assert stale.terminate.call_count == 1
        assert module.pyaudio.PyAudio.call_count == 2
        assert recorder.pyaudio is rebuilt
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_the_rebuilding_press_is_not_reported_microphone_busy(
    audio_recorder_module, coreaudio_stub
):
    """The symptom of the discarded take: the take was empty, so
    ``finish_recording`` took the warm-not-ready branch and wrote
    MICROPHONE_BUSY for a reset that had succeeded -- blaming a microphone
    that nothing was holding. Under the fix the press never enters that
    branch, because it is not a warm press."""
    module = audio_recorder_module
    release = threading.Event()
    stale = _make_pyaudio_instance(0, rate=16000.0)
    stale.open.return_value = _streaming_stream(release)
    rebuilt = _make_pyaudio_instance(5, rate=16000.0)
    rebuilt.open.return_value = _streaming_stream(release)
    recorder = _make_recorder(module, init_probe=stale, extra_pa=rebuilt)
    recorder.set_keep_warm(True)
    assert _wait_until(lambda: recorder._warm.ready)
    _live_default_input(coreaudio_stub, 77)

    try:
        assert recorder.start_recording() is True
        # Do not wait for chunks: stop fast, the way a short press would.
        # A press routed into the not-yet-open capture reports busy here.
        recorder.stop_recording()
        assert recorder.last_error is None, (
            f"the rebuilding press was mislabelled: {recorder.last_error!r}"
        )
    finally:
        release.set()
        _cleanup_recorder(recorder)
