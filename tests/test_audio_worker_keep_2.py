"""A worker that was noted but never started is ARMING, not finished.

``start_recording()`` notes the worker under ``_lock`` and starts it after
the release -- a window in which the thread exists, has never run, and may
be about to enter ``pa.open()``. ``WorkerKeep`` pruned on ``is_alive()``,
which is False for a thread that has not started: the next ``note()``
erased the pending worker from the ledger and the terminate it was there
to block sailed through underneath it.

Mirror of the warm-side rule (``WarmKeep.installed_may_be_native``): no
thread yet / not yet started == arming == may be native. Pruning applies
only to threads that STARTED and finished. A thread whose ``start()``
itself raised can never enter PortAudio, so the launch path must take it
back out of the ledger explicitly -- an unstarted thread is now a blocker,
and a leaked one would refuse every later ``terminate_all()`` for the life
of the process.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from kuiskaus.audio_device import PyAudioRegistry
from kuiskaus.audio_worker_keep import WorkerKeep
from tests.support_audio_recorder import _cleanup_recorder, _make_recorder


def _registry(pa_module=None):
    """A registry over a stubbed pyaudio module yielding one fresh mock
    instance per construction."""
    module = pa_module or MagicMock(name="pyaudio-module")
    pa = MagicMock(name="PyAudioInstance")
    module.PyAudio = MagicMock(return_value=pa)
    return PyAudioRegistry(module), pa, module


def _live_thread() -> tuple[threading.Event, threading.Thread]:
    hold = threading.Event()
    thread = threading.Thread(target=hold.wait, daemon=True)
    thread.start()
    return hold, thread


def _finished_thread() -> threading.Thread:
    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    return thread


# ---------------------------------------------------------------------------
# The arming window: noted, never started
# ---------------------------------------------------------------------------


def test_an_unstarted_noted_worker_blocks_terminate_all():
    """The gate must read "may be native" conservatively: the recorder
    notes under the lock and starts after the release, so a terminate
    racing that window sees a thread whose start() may be next on the
    CPU -- about to reach pa.open() on the instance being terminated."""
    registry, pa, _module = _registry()
    registry.open(lambda _pa: 3)
    never_started = threading.Thread(target=lambda: None, daemon=True)
    registry.note_worker(never_started)

    assert registry.terminate_all() is False
    pa.terminate.assert_not_called()


def test_note_keeps_a_thread_that_has_never_started():
    """Pruning keyed on is_alive() erased the pending worker: a second
    note() in the window between note and start dropped it from the very
    list that exists to name it."""
    keep = WorkerKeep()
    pending = threading.Thread(target=lambda: None, daemon=True)
    finished = _finished_thread()
    keep.note(finished)
    keep.note(pending)
    hold, later = _live_thread()
    try:
        keep.note(later)  # the prune sweep
        assert pending in keep._workers, (
            "the not-yet-started worker was pruned as 'finished' -- it is "
            "arming, and its start() may be the next native call"
        )
        assert finished not in keep._workers
        assert keep.any_may_be_native() is True
    finally:
        hold.set()
        later.join(timeout=5.0)
        keep.forget(pending)


def test_any_may_be_native_reports_an_unstarted_worker():
    keep = WorkerKeep()
    keep.note(threading.Thread(target=lambda: None, daemon=True))
    assert keep.any_may_be_native() is True


def test_a_started_and_finished_worker_prunes_and_releases_the_gate():
    """The prune rule survives the conservative reading: a worker that
    started and finished is provably out of PortAudio and must not
    accumulate or keep blocking terminates."""
    registry, pa, _module = _registry()
    registry.open(lambda _pa: 3)
    finished = _finished_thread()
    registry.note_worker(finished)

    assert registry.terminate_all() is True
    pa.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# The failed launch: a start() that raised must not linger as a blocker
# ---------------------------------------------------------------------------


def test_start_recording_rolls_back_a_worker_whose_start_raised(
    audio_recorder_module,
):
    """End to end through the press path: a thread-limit storm makes the
    worker's start() fail. The press must not leave a phantom behind --
    neither a ledger entry that refuses every later terminate_all() for
    the life of the process, nor a recording/recording_thread pair that
    stop_recording() would try to JOIN (joining an unstarted thread
    raises RuntimeError)."""
    module = audio_recorder_module
    real_thread = threading.Thread

    class _StormThread(real_thread):
        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    class _ThreadNamespace:
        Thread = _StormThread

    recorder = _make_recorder(module)
    real_threading = module.threading
    try:
        module.threading = _ThreadNamespace()
        with pytest.raises(RuntimeError):
            recorder.start_recording()
        module.threading = real_threading

        assert recorder._session.workers._workers == []
        assert recorder.recording is False
        assert recorder.recording_thread is None
        # stop_recording() would raise joining an unstarted thread; the
        # rolled-back state returns the empty take instead.
        assert recorder.stop_recording().size == 0
        assert recorder._session.terminate_all() is True
    finally:
        module.threading = real_threading
        _cleanup_recorder(recorder)
