"""Who may still be inside a native PortAudio call.

Two ledger gaps, both caught by the harness of the first fix:

1. workers had no ledger. ``terminate_all()`` consulted only
   the named worker plus the keep-warm captures, and ``recording_thread`` is
   a *name*, cleared by whichever successor completes first. A thread that
   lost its name but was still inside ``pa.open()`` was invisible to the
   gate, and the next press terminated PortAudio underneath it. ``WorkerKeep``
   is the ledger: every worker the session started stays named until its
   thread has actually finished. That scenario itself is reproduced in
   ``tests/test_audio_recorder_8.py``.

2. **ISSUE** -- ``terminate_all()`` asked only the *dropped* captures. A
   capture still installed in ``WarmArm`` (the worker path never hands one
   over, resting on "with keep-warm on there is no worker", which is a race)
   was not a blocker at all. Covered by ``WarmKeep.note_installed`` /
   ``installed_may_be_native``.

Both are tested against ``PyAudioRegistry.terminate_all()`` -- the call that
actually pulls PortAudio out -- not just against the gate that reads them.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from kuiskaus.audio_warm_keep import WarmKeep
from kuiskaus.audio_worker_keep import WorkerKeep
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input


def _capture(thread=None, *, stops=True, released_after=None):
    """A ``WarmCapture`` stand-in. All ``WarmKeep`` needs of an installed
    capture is its reader thread; of a dropped one, its ``stop()``.

    Built on ``MagicMock`` rather than a duck-typed class so the type checker
    reads it as the real thing at the typed ``note_installed``/``drop``
    boundaries. ``thread`` is the only thing the ledger asks of an installed
    capture, and it asks ``thread.is_alive()`` -- the same question the real
    ``WarmCapture`` answers.
    """
    warm = MagicMock(name="warm-capture")
    warm.thread = thread

    def stop(timeout: float = 2.0) -> bool:
        if timeout == 0.0 and released_after is not None:
            # The non-blocking probe a second drop asks for: free once the
            # wedged read's thread is gone.
            return not released_after.is_alive()
        return stops

    warm.stop.side_effect = stop
    return warm


def _live_thread() -> tuple[threading.Event, threading.Thread]:
    hold = threading.Event()
    thread = threading.Thread(target=hold.wait, daemon=True)
    thread.start()
    return hold, thread


# ---------------------------------------------------------------------------
# The worker ledger
# ---------------------------------------------------------------------------


def test_worker_keep_names_a_live_thread_and_forgets_a_finished_one():
    keep = WorkerKeep()
    hold, worker = _live_thread()
    keep.note(worker)
    assert keep.any_may_be_native() is True

    hold.set()
    worker.join(timeout=5.0)
    assert keep.any_may_be_native() is False


def test_worker_keep_never_counts_the_thread_that_is_asking():
    """The thread running a reset joined its predecessor first, so it is not
    its own blocker -- the same rule the named-worker gate already had."""
    keep = WorkerKeep()
    keep.note(threading.current_thread())
    assert keep.any_may_be_native() is False


def test_worker_keep_prunes_finished_threads_so_the_ledger_stays_bounded():
    keep = WorkerKeep()
    hold, first = _live_thread()
    hold.set()
    first.join(timeout=5.0)
    keep.note(first)
    hold2, second = _live_thread()
    keep.note(second)
    try:
        # The finished thread is pruned on insert, so the list does not grow
        # for the life of the process.
        assert len(keep._workers) == 1
        assert keep.any_may_be_native() is True
    finally:
        hold2.set()
        second.join(timeout=5.0)


def test_terminate_all_refuses_under_a_worker_the_caller_never_named(coreaudio_stub):
    """The core invariant, tested at the call that terminates.

    A worker whose ``recording_thread`` name was taken by a successor is
    still a live thread that may be inside ``pa.open()``. The registry has to
    know it anyway, or the press that resets drives PortAudio's init count to
    zero underneath it.
    """
    registry, pa, _module = _registry()
    registry.open(_find_device)
    hold, orphan = _live_thread()
    registry.note_worker(orphan)
    # The name a caller could pass is gone (a successor completed and cleared
    # it); only the ledger still remembers the thread.
    try:
        assert registry.terminate_all() is False
        pa.terminate.assert_not_called()
    finally:
        hold.set()
        orphan.join(timeout=5.0)

    # Once the thread is gone the gate lifts on its own -- nothing to wait for.
    assert registry.terminate_all() is True
    pa.terminate.assert_called_once_with()


def test_worker_reset_reports_busy_for_a_worker_no_caller_named(coreaudio_stub):
    """Same invariant on the reset's own gate, which is where a press decides
    whether it may terminate at all."""
    registry, pa, _module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    hold, orphan = _live_thread()
    registry.note_worker(orphan)
    try:
        result = registry.refresh(_find_device)
        assert result.busy is True
        pa.terminate.assert_not_called()
        assert _module.PyAudio.call_count == 1  # nothing rebuilt
    finally:
        hold.set()
        orphan.join(timeout=5.0)

    assert registry.refresh(_find_device).busy is False
    assert pa.terminate.call_count == 1
    assert _module.PyAudio.call_count == 2  # ... and now it is rebuilt


# ---------------------------------------------------------------------------
# The capture installed in the arm, not just the ones already dropped
# ---------------------------------------------------------------------------


def test_terminate_all_refuses_while_an_installed_capture_may_be_reading():
    """``WarmKeep._blockers`` only ever
    held captures that had been *dropped*. The worker path hands no capture
    at all, resting on "with keep-warm on there is no worker" -- a race, not
    an invariant: a capture armed from the menu during a press reads on a
    registered instance while terminate_all() drives the count to zero."""
    registry, pa, _module = _registry()
    registry.open(_find_device)
    hold, reader = _live_thread()
    registry.warm.note_installed(_capture(thread=reader))
    try:
        assert registry.terminate_all() is False
        pa.terminate.assert_not_called()
    finally:
        hold.set()
        reader.join(timeout=5.0)

    assert registry.terminate_all() is True
    pa.terminate.assert_called_once_with()


def test_an_installed_capture_with_no_thread_yet_still_blocks_a_terminate():
    """The gate was blind to
    the one window ``note_installed`` exists to cover.

    ``WarmArm`` names the capture BEFORE ``capture.start()`` assigns its
    thread, precisely so a terminate racing the arm sees it -- but
    ``terminate_all()`` takes no recorder lock, so it can read the ledger in
    that gap, where the predicate's ``thread is not None`` conjunct answered
    "no blocker" for a capture that is arming, not idle. Measured before the
    fix: with the capture installed and its thread still unassigned,
    ``installed_may_be_native()`` said False, ``terminate_blocker()`` said
    None, and ``Pa_Terminate`` fired.
    """
    registry, pa, _module = _registry()
    registry.open(_find_device)
    arming = _capture(thread=None)  # installed, not started yet
    registry.warm.note_installed(arming)

    assert registry.terminate_all() is False
    pa.terminate.assert_not_called()

    # A gate, not a verdict: once start() has assigned a thread that has
    # provably finished, the same session terminates.
    done = threading.Event()
    reader = threading.Thread(target=done.wait, daemon=True)
    arming.thread = reader
    reader.start()
    done.set()
    reader.join(timeout=5.0)

    assert registry.terminate_all() is True
    pa.terminate.assert_called_once_with()


def test_installed_capture_is_never_its_own_blocker():
    """The capture's own thread runs the reset from ``_open_warm_stream``. If
    it counted itself the keep-warm session could never reset at all."""
    keep = WarmKeep()
    keep.note_installed(_capture(thread=threading.current_thread()))
    assert keep.installed_may_be_native() is False


def test_a_stopped_installed_capture_stops_blocking():
    """drop() is the hand-off: the capture leaves the installed slot and the
    dropped ledger decides from then on, so a cleanly stopped capture does
    not hold the engine open forever."""
    keep = WarmKeep()
    hold, reader = _live_thread()
    capture = _capture(thread=reader)
    keep.note_installed(capture)
    assert keep.installed_may_be_native() is True

    try:
        assert keep.drop(capture) is True
        assert keep.installed_may_be_native() is False
        assert keep.clear() is True
    finally:
        hold.set()
        reader.join(timeout=5.0)


def test_an_installed_capture_that_would_not_stop_stays_a_blocker():
    """The wedged case: dropped for presses (its stop event is set, so it can
    never re-open) but still reading, so it must remain visible to every
    later terminate_all()."""
    keep = WarmKeep()
    hold, reader = _live_thread()
    capture = _capture(thread=reader, stops=False, released_after=reader)
    keep.note_installed(capture)
    try:
        assert keep.drop(capture) is False
        assert keep.installed_may_be_native() is False  # out of the slot...
        assert keep.clear() is False  # ... and in the blockers list instead
    finally:
        hold.set()
        reader.join(timeout=5.0)
    assert keep.clear() is True
