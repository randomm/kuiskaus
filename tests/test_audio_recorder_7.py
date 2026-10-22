"""The stuck keep-warm capture: what it blocks, and what brings it back.

Split out of test_audio_recorder_3.py, which had grown past the 500-line
module cap (issue #72 review). These are the recorder-level lifecycle tests
for a keep-warm capture whose thread is parked in a native
``stream.read()``: the capture comes out of the recorder's state so later
presses do not record into a dead sink, but its thread is still alive and
still reading, and nothing may drive PortAudio's init count to zero
underneath it until that thread has actually exited.
"""

import threading
from unittest.mock import MagicMock

from kuiskaus.audio_warm_keep import WarmKeep
from tests.support_audio_recorder import (
    _blocking_stream,
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
    _warm_recorder,
)


def _stuck_warm_recorder(module, coreaudio_stub):
    """A recorder whose keep-warm capture is ready and can only be "stopped"
    the way ``WarmCapture.stop()`` behaves on a stuck read -- set the stop
    event, report failure -- with its real thread parked in the read until
    ``release`` is set. The live default is moved so the next press resets.

    Returns ``(recorder, pa, release)``; ``pa`` is the single live instance
    every open in the test lands on, and no second instance is available --
    a reset that decided to rebuild would surface as a construction error.
    """
    release = threading.Event()

    def _blocked_open(*_args, **_kwargs):
        # read() blocks on `release` and then raises: the capture's thread is
        # deterministically parked inside the native call, which is the only
        # state where stop() reports failure for real.
        return _blocking_stream(OSError("stream closed by test"), release)

    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.side_effect = _blocked_open
    # The instance a later, legitimate reset rebuilds with -- also blocking,
    # so a worker on it cannot spin on an unconstrained mock read.
    rebuild = _make_pyaudio_instance(5, rate=16000.0)
    rebuild.open.side_effect = _blocked_open
    recorder = _make_recorder(module, init_probe=pa, extra_pa=rebuild)
    recorder.set_keep_warm(True)
    assert _wait_until(lambda: recorder._warm.ready)

    warm = recorder._warm

    def _stuck_stop(timeout: float = 2.0) -> bool:
        # What the real stop() does: set the stop event, join for `timeout`,
        # and report failure while the thread is parked in the read.
        warm._stop.set()
        warm._thread.join(timeout)
        return not warm._thread.is_alive()

    warm.stop = _stuck_stop
    _live_default_input(coreaudio_stub, 77)
    return recorder, pa, release


def test_busy_warm_press_drops_the_capture_it_could_not_stop(
    audio_recorder_module, coreaudio_stub
):
    """AC #4's stuck-warm scenario, continued: warm.stop() False fails the
    press busy, but the capture is ALSO dead-on-arrival -- stop() set its
    stop event, so when the stuck read returns the thread exits and that
    WarmCapture can never re-open. The busy press must drop it: keeping
    it installed lets the next press that needs no reset record silently
    into a dead sink while keep-warm stays advertised (#72 review)."""
    module = audio_recorder_module
    release = threading.Event()
    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.return_value = _streaming_stream(release)
    # The reset's rebuild instance: its stream dies with `release` too,
    # so the follow-through worker ends deterministically.
    rebuild = _make_pyaudio_instance(5, rate=16000.0)
    rebuild.open.return_value = _streaming_stream(release)
    recorder = _make_recorder(module, init_probe=pa, extra_pa=rebuild)
    recorder.set_keep_warm(True)
    assert _wait_until(lambda: recorder._warm.ready)
    warm = recorder._warm

    def _stuck_stop(_timeout: float = 2.0) -> bool:
        warm._stop.set()  # the real stop() sets the event before it times out
        return False

    warm.stop = _stuck_stop
    _live_default_input(coreaudio_stub, 77)  # the default moved

    assert recorder.start_recording() is False  # busy
    assert "busy" in recorder.last_error.lower()
    assert recorder._warm is None  # the dead capture is gone
    pa.terminate.assert_not_called()  # engine still up: nothing terminated
    assert module.pyaudio.PyAudio.call_count == 1  # ... and nothing rebuilt

    # The next press takes the worker path -- not warm.begin() on a sink
    # that can no longer deliver.
    assert recorder.start_recording() is True
    worker = recorder.recording_thread
    assert worker is not None  # a per-press worker, not the warm route
    release.set()  # every mock read raises OSError from here: end the worker
    assert _wait_until(lambda: not worker.is_alive())


def test_busy_warm_press_keeps_a_capture_that_still_works(
    audio_recorder_module, coreaudio_stub
):
    """The drop belongs to the warm-stop-False path alone: a
    blocked_worker busy report never stopped the capture, so keep-warm
    survives that press intact (#72 review)."""
    module = audio_recorder_module
    release = threading.Event()
    recorder, pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)
    warm = recorder._warm

    blocker = threading.Event()
    stalled = threading.Thread(target=blocker.wait, daemon=True)
    stalled.start()
    recorder.recording_thread = stalled  # orphan the press must not reset under
    _live_default_input(coreaudio_stub, 77)
    try:
        assert recorder.start_recording() is False  # busy: orphan still alive
    finally:
        blocker.set()
        stalled.join(timeout=5.0)

    assert recorder._warm is warm  # never stopped -- never dropped
    pa.terminate.assert_not_called()
    release.set()


def test_press_after_a_dropped_warm_capture_does_not_terminate_under_it(
    audio_recorder_module, coreaudio_stub
):
    """CRITICAL (issue #72 review): dropping the stuck capture from the
    recorder must not orphan its live thread.

    The abandoned thread is alive by definition -- that is exactly why
    stop() timed out -- with its stream still open on the registered
    instance. Before the fix the recorder's ``_warm = None`` was the only
    handle on it, so press 2 saw ``warm is None``, believed PortAudio was
    clear, and drove the init count to zero with a reader still parked
    inside ``stream.read()``. It must report busy instead: nothing
    terminated, nothing rebuilt, and the press still records -- on the live
    instance, through the worker path.
    """
    recorder, pa, release = _stuck_warm_recorder(audio_recorder_module, coreaudio_stub)
    warm_thread = recorder._warm._thread
    assert warm_thread is not None and warm_thread.is_alive()

    try:
        assert recorder.start_recording() is False  # press 1: busy, capture dropped
        assert recorder._warm is None
        assert warm_thread.is_alive()  # alive by definition of the stuck stop
        assert pa.terminate.call_count == 0

        assert recorder.start_recording() is True  # press 2: the worker path
        worker = recorder.recording_thread
        assert worker is not None
        # The gate the prior fix had no way to check: still nothing terminated
        # and nothing rebuilt while that reader sits in the native call.
        assert pa.terminate.call_count == 0
        assert audio_recorder_module.pyaudio.PyAudio.call_count == 1
        release.set()
        assert _wait_until(lambda: not worker.is_alive())
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_cleanup_does_not_terminate_under_a_dropped_warm_capture(
    audio_recorder_module, coreaudio_stub
):
    """The quit path holds the same invariant as a press, and had the same
    blind spot: cleanup() only asked the capture it still had. With the stuck
    capture already dropped from ``_warm`` that guard was vacuously true, so
    quitting ran PyAudio.terminate() under a thread still inside
    stream.read() -- the one state cleanup() exists to refuse (issue #37
    task-c). The dropped capture's handle lives on the session now, so
    cleanup() has to ask there too.
    """
    recorder, pa, release = _stuck_warm_recorder(audio_recorder_module, coreaudio_stub)
    warm_thread = recorder._warm._thread
    assert warm_thread is not None

    try:
        assert recorder.start_recording() is False  # press 1 drops the capture
        assert recorder._warm is None
        assert pa.terminate.call_count == 0

        recorder.cleanup()

        assert pa.terminate.call_count == 0, (
            "cleanup() terminated PortAudio while a thread it had forgotten "
            "was still inside stream.read()"
        )
        release.set()
        assert _wait_until(lambda: not warm_thread.is_alive(), timeout=8.0)
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_keep_warm_returns_after_a_dropped_capture_thread_leaves_the_read(
    audio_recorder_module, coreaudio_stub
):
    """Pre-roll comes back on its own (issue #72 review).

    ``rearm_warm(True)`` was otherwise reachable only from a warm press that
    stopped its capture, so one stuck read left keep-warm off for the life of
    the process while the menu still showed the checkmark -- every later
    press paid worker-path open latency for a feature the user never turned
    off. Once the abandoned thread has exited nothing blocks arming a fresh
    capture, and the next worker press arms it.
    """
    recorder, _pa, release = _stuck_warm_recorder(audio_recorder_module, coreaudio_stub)
    warm_thread = recorder._warm._thread

    try:
        assert recorder.start_recording() is False  # press 1 drops the capture
        assert recorder._warm is None
        release.set()  # let the abandoned read return so its thread can exit
        assert _wait_until(lambda: not warm_thread.is_alive(), timeout=8.0)

        assert recorder.start_recording() is True  # a worker press
        assert _wait_until(lambda: recorder._warm is not None), (
            "keep-warm was never re-armed after the stuck read cleared"
        )
        # Nothing blocks the deferred reset any more either: the abandoned
        # thread is gone, so terminating and rebuilding is what the session
        # is supposed to do here. The worker may have finished by the time
        # this reads the name (every read in this test releases instantly
        # once `release` is set), so the assertion covers "ran and cleared
        # its own state", not "is still running".
        assert _wait_until(lambda: not recorder.recording, timeout=8.0), (
            "the worker press never completed its own teardown"
        )
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_worker_path_does_not_rearm_warm_when_the_user_switched_it_off(
    audio_recorder_module, coreaudio_stub
):
    """Recovery honours the menu: a capture the user unchecked must not come
    back on some later press just because a thread finally unwedged.
    """
    recorder, _pa, release = _stuck_warm_recorder(audio_recorder_module, coreaudio_stub)
    warm_thread = recorder._warm._thread

    try:
        assert recorder.start_recording() is False  # press 1 drops the capture
        recorder.set_keep_warm(False)  # the user gives up on keep-warm
        release.set()
        assert _wait_until(lambda: not warm_thread.is_alive(), timeout=8.0)

        assert recorder.start_recording() is True
        worker = recorder.recording_thread
        assert worker is not None
        assert _wait_until(lambda: not worker.is_alive(), timeout=8.0)
        assert recorder._warm is None, "keep-warm came back after the user unchecked it"
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_refused_warm_press_joins_a_wedged_capture_only_once(
    audio_recorder_module, coreaudio_stub
):
    """Issue #72 review, round 2 (measured regression): ONE refused press
    joined the wedged keep-warm thread TWICE -- refresh() dropped the capture,
    then handed the same object to rearm_warm(False), which dropped it again.
    Each stop() is a 2.0s join of a thread proven parked in stream.read(), so
    a press that records nothing stalled the hotkey dispatcher for ~4s: the
    press-path stall class issue #60 exists to keep off this thread. The press
    now joins once and every later ask is a non-blocking probe.
    """
    module = audio_recorder_module
    release = threading.Event()
    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.return_value = _blocking_stream(OSError("stream closed by test"), release)
    recorder = _make_recorder(module, init_probe=pa)
    recorder.set_keep_warm(True)
    assert _wait_until(lambda: recorder._warm.ready)
    warm = recorder._warm
    timeouts: list[float] = []

    def _stuck_stop(timeout: float = 2.0) -> bool:
        # What the real stop() reports while the thread is parked in the
        # read. The join itself is skipped -- the assertion is about how many
        # BLOCKING joins one refused press asks for.
        timeouts.append(timeout)
        warm._stop.set()
        return False

    warm.stop = _stuck_stop
    _live_default_input(coreaudio_stub, 77)  # the default moved

    try:
        assert recorder.start_recording() is False  # refused: microphone busy
        blocking = [t for t in timeouts if t != 0.0]
        assert blocking == [2.0], f"blocking joins for one refused press: {blocking}"
        assert recorder._warm is None  # the dead capture was handed back
        pa.terminate.assert_not_called()  # ... and the engine stayed up
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_warm_keep_a_second_drop_of_a_wedged_capture_probes_never_rejoins():
    """WarmKeep unit (issue #72 review, round 2): the busy reset path drops a
    wedged capture and then hands the SAME object to rearm_warm(False), which
    drops it again. A capture already on the blockers list has been joined and
    is proven parked, so the second ask is answered by the probe -- two joins
    of a thread that cannot answer is 2s of stall per ask, on the press path.
    """
    keep = WarmKeep()
    wedged = MagicMock(name="wedged")
    wedged.stop.return_value = False  # never stops: parked in the native read

    assert keep.drop(wedged) is False
    assert keep.drop(wedged) is False  # the rearm_warm(False) hand-back

    # First ask: the real stop() with WarmCapture's own default timeout; the
    # hand-back: the non-blocking probe. A regression shows up as [2.0, 2.0].
    timeouts = [c.args[0] if c.args else 2.0 for c in wedged.stop.call_args_list]
    assert timeouts == [2.0, 0.0], f"a wedged capture was joined twice: {timeouts}"
    assert keep.clear() is False  # still a blocker: terminate stays refused


def test_warm_keep_recovers_only_once_its_thread_is_gone():
    """WarmKeep unit: the handle is what makes the invariant checkable, and it
    is released by a non-blocking probe -- never by a join that could park the
    caller on a thread that may be parked for the life of the process.
    """
    keep = WarmKeep()
    rearm = MagicMock()
    stuck = MagicMock()
    stuck.stop.side_effect = [False, False, False, True]
    keep.want(True)  # the user asked for keep-warm before any capture wedged

    assert keep.drop(stuck) is False
    assert keep.clear() is False  # still parked -> terminate must be refused
    keep.recover(rearm)
    rearm.assert_not_called()  # not yet: recover() probes, and it is still reading

    assert keep.clear() is True  # the read returned, so the thread exited
    keep.recover(rearm)
    rearm.assert_called_once_with(True)
    keep.recover(rearm)  # already armed -- a second re-arm would replace it
    rearm.assert_called_once_with(True)
    # Every probe after the drop is non-blocking, whatever the capture's own
    # stop() default is.
    assert [c.args for c in stuck.stop.call_args_list[1:]] == [
        (0.0,),
        (0.0,),
        (0.0,),
    ]


def test_warm_keep_ignores_a_capture_that_stopped_cleanly():
    """WarmKeep unit: a capture that stopped has nothing left to block on, and
    is not a recovery candidate -- that press re-arms through the reset path.
    """
    keep = WarmKeep()
    rearm = MagicMock()
    stopped = MagicMock()
    stopped.stop.return_value = True

    assert keep.drop(stopped) is True
    assert keep.clear() is True
    keep.want(True)
    keep.recover(rearm)
    rearm.assert_not_called()


def test_warm_keep_want_off_withdraws_a_pending_recovery():
    """WarmKeep unit: unchecking keep-warm while a capture is wedged must not
    leave a recovery armed for some later press -- but the wedged thread still
    blocks termination.
    """
    keep = WarmKeep()
    rearm = MagicMock()
    stuck = MagicMock()
    stuck.stop.return_value = False
    keep.want(True)
    assert keep.drop(stuck) is False

    keep.want(False)
    assert keep.clear() is False  # still must refuse to terminate underneath it
    keep.recover(rearm)
    rearm.assert_not_called()


def test_warm_keep_a_probe_during_the_join_sees_the_capture_being_stopped():
    """WarmKeep unit (issue #72 review, round 1): the blocker must be named
    BEFORE ``stop()`` joins.

    The join window is exactly the interval in which the capture's thread is
    proven to be parked in ``stream.read()``. Registering only after
    ``stop()`` returned left the capture invisible there, so a ``clear()``
    racing in from the rumps main thread (set_keep_warm and cleanup() run
    there, while a stuck press runs on the hotkey dispatcher's worker -- and
    CallbackDispatcher.stop() tolerates that callback outliving its join)
    reported a session it was free to terminate, with a reader still parked
    inside PortAudio.
    """
    keep = WarmKeep()
    entering_the_join = threading.Event()
    let_the_join_finish = threading.Event()
    probe_saw = {}

    wedged = MagicMock(name="wedged-capture")

    def _wedged_stop(timeout: float = 2.0) -> bool:
        if timeout == 0.0:
            return False  # a probe: the thread is still in the read
        entering_the_join.set()
        let_the_join_finish.wait(5.0)  # the blocking join itself
        return False  # and it never came back

    wedged.stop.side_effect = _wedged_stop

    def probe():
        entering_the_join.wait(5.0)
        probe_saw["clear"] = keep.clear()

    observer = threading.Thread(target=probe, daemon=True)
    observer.start()
    try:
        assert keep.drop(wedged) is False
        observer.join(timeout=5.0)
        assert probe_saw == {"clear": False}, (
            "a concurrent clear() reported a session it was free to "
            "terminate while drop() was mid-join on the capture blocking it"
        )
    finally:
        let_the_join_finish.set()
        observer.join(timeout=5.0)


def test_warm_keep_clear_prunes_rather_than_reassigning_the_blocker_list():
    """WarmKeep unit (issue #72 review, round 1): a capture registered while
    ``clear()``'s probes were running must survive the commit.

    The probes cannot hold the lock (a wedged capture must never be able to
    hold up the other thread's probe), so a ``drop()`` landing between the
    snapshot and the commit is a lost update if the commit writes the probed
    list back. Losing an entry permanently un-gates every later
    terminate_all() against a thread still inside a native read.
    """
    keep = WarmKeep()
    late = MagicMock(name="late-capture")
    late.stop.return_value = False  # stuck: it registers as a blocker

    retiring = MagicMock(name="retiring-capture")

    def _retiring_stop(timeout: float = 2.0) -> bool:
        # The other thread's drop() lands inside this probe.
        if timeout == 0.0:
            keep.drop(late)
            return True  # and this capture's thread has really exited
        return False

    retiring.stop.side_effect = _retiring_stop
    assert keep.drop(retiring) is False  # wedged -> it is a blocker now

    assert keep.clear() is False, (
        "clear() erased a blocker another thread registered mid-probe"
    )
    assert late in keep._blockers
    assert retiring not in keep._blockers  # the provably-dead one WAS pruned
    keep.clear()  # and from here on the survivor is probed like any blocker
    late.stop.assert_called_with(0.0)
