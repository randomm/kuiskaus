"""Recorder transitions and gates that must be serialised under _lock.

Split out of ``audio_recorder`` (at its 500-line cap) for issue #72. Every
helper here decides or mutates recorder state that has more than one caller
thread, and each hazard was a window where that state could change between
the check and the act:

* the keep-warm capture was armed by a lock-free check-then-act, and
  #72 gave arming two extra caller threads (the worker's
  ``warm.recover(self._rearm_warm)`` hand-back, a hotkey press's
  ``rearm_warm(True)`` after a device reset). Two threads could both
  pass the ``_warm is None`` check and both start a capture; the loser
  is referenced by nobody and absent from ``WarmKeep._blockers``, so
  ``terminate_all()`` drives PortAudio out from under it. ``WarmArm``
  makes arm/drop one atomic transition under the recorder's own lock
  (passed in, not duplicated: the capture is recorder state and every
  other reader holds that lock), install and thread-spawn included -- a
  concurrent ``take()`` can never slip in between them.
* the same re-arm forced keep-warm back ON. A reset runs outside the
  recorder lock for as long as a full teardown and rebuild, and its
  hand-back went through ``set_keep_warm(True)``, which records the
  request as well as satisfying it -- so a menu uncheck landing inside
  that window was erased, the capture re-armed, and the mic left open
  behind an unchecked menu item. ``WarmArm.rearm`` satisfies the
  request the user actually has, read under the same lock as the arm
  (``set_enabled`` -- the menu's path -- takes that lock too, so the
  check and the act cannot be split by a click).
* the reset gate trusted a worker name captured before a join of up to
  30 s; by gate time that worker may have finished and a live one
  become current. ``worker_may_be_native`` resolves the names through
  a callable at gate time, and ``warm_press_gate`` snapshots the press
  state under the lock instead of reading three fields one by one.

Lock discipline (issue #60): no native call -- PortAudio open/terminate,
``thread.start()``, ``thread.join()`` -- ever runs under the recorder
lock from this module.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from kuiskaus.audio_resample import assemble_audio
from kuiskaus.audio_retry import (
    PREVIOUS_WORKER_WAIT_SECONDS,
    find_default_input_device,
    log_stop,
    open_warm_stream,
    refresh_pyaudio_session,
    worker_alive,
)

if TYPE_CHECKING:
    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry
    from kuiskaus.audio_recorder import AudioRecorder
    from kuiskaus.audio_warm import WarmCapture
    from kuiskaus.audio_warm_keep import WarmKeep


class WarmArm:
    """The keep-warm capture slot, armed and dropped atomically.

    Threading: ``_lock`` IS the recorder's ``_lock``. ``make_capture``
    and ``note_capture_rate`` run under it (WarmCapture's constructor is
    pure Python -- no native PortAudio); ``thread.start()`` runs under
    it too -- a spawn is not a PortAudio call, and it keeps the
    install-and-spawn one atomic step, so a concurrent ``take()`` can
    never slip between "still installed" and "started" and leave a
    live thread that nothing on the blockers list can name (the child
    thread itself reaches ``recorder._lock`` only after the transition
    lock is released). ``WarmKeep.drop()`` runs after the release: its
    join can take seconds.
    """

    def __init__(self, keep: WarmKeep, lock: threading.Lock) -> None:
        self._keep = keep
        self._lock = lock
        self.capture: WarmCapture | None = None

    def set_enabled(
        self,
        make_capture: Callable[[], WarmCapture],
        note_capture_rate: Callable[[int | None], None],
        enabled: bool,
    ) -> None:
        """The menu's request (issue #66): record it, then bring the capture
        to match it. At most one capture is ever started per call, and a
        capture that lost the slot to a concurrent take() is never started
        at all."""
        self._keep.want(enabled)
        dropped: WarmCapture | None = None
        with self._lock:
            if enabled and self.capture is None:
                self._arm_locked(make_capture)
            elif not enabled and self.capture is not None:
                dropped, self.capture = self.capture, None
                note_capture_rate(dropped.capture_rate)
        if dropped is not None:
            # drop() is what clears the installed name, in the same step that
            # decides whether the thread still blocks a terminate.
            self._keep.drop(dropped)

    def rearm(self, make_capture: Callable[[], WarmCapture]) -> None:
        """Bring the capture back IF the user still wants keep-warm.

        The reset's and the recovery's hand-back used to go through
        ``set_enabled(True)``, which also WRITES the request -- so a menu
        uncheck landing during the reset (which runs outside this lock, for
        as long as a teardown and rebuild takes) was silently reversed and the
        microphone left open behind an unchecked menu item (issue #72). This path satisfies the request; it never makes
        one. The request is read under this same lock, so a concurrent menu
        click cannot land between the check and the arm.
        """
        with self._lock:
            if self.capture is not None:
                # Already armed for the request: any pending recovery is
                # satisfied by the capture that is installed.
                self._keep.note_armed()
            elif self._keep.wanted():
                self._arm_locked(make_capture)

    def _arm_locked(self, make_capture: Callable[[], WarmCapture]) -> None:
        """Install and start one capture. Caller holds ``_lock``."""
        capture = make_capture()
        self.capture = capture
        # Named for the session BEFORE it is started: a terminate racing this
        # transition must see the capture whose thread it is about to read
        # on, and after the release there is no such window -- WarmKeep holds
        # the name either way.
        self._keep.note_installed(capture)
        try:
            capture.start()
        except BaseException:
            # A capture that never started can never read, so it must not
            # stay named as a blocker: installed_may_be_native() treats an
            # installed capture with no thread as arming, and a leaked name
            # here would refuse every later terminate_all() for the life of
            # the process (thread creation does fail -- RuntimeError under a
            # thread-limit storm). Undo the install and let the caller see
            # the failure, as it always has.
            self.capture = None
            self._keep.note_installed(None)
            # ... and ask for recovery: the user's request is still live
            # and the slot is empty, and nothing else ever re-arms from
            # this state (the reset's hand-back only fires after a reset),
            # so without the ask keep-warm stayed OFF behind a checked
            # menu for the life of the process (issue #72).
            # The worker path's recover() picks it up on a later press.
            self._keep.need_recovery()
            raise
        self._keep.note_armed()

    def take(self) -> WarmCapture | None:
        """Clear the slot and hand the capture to the caller (rearm /
        cleanup); the caller owes it a ``WarmKeep.drop()`` so its thread
        stays nameable if it is parked in a native read."""
        with self._lock:
            dead, self.capture = self.capture, None
            return dead


def start_press(
    recorder: AudioRecorder,
    warm: WarmCapture | None,
    sink: queue.Queue[bytes],
    my_gen: int,
    worker: threading.Thread | None,
) -> None:
    """Launch the admitted press: route the ring to the warm capture, or
    start the worker (issue #66 / #72).

    ``worker`` is the thread the press built and noted, handed in by the
    caller rather than re-read from ``recorder.recording_thread`` here: the
    ledger counts a noted-but-never-started thread as a blocker, so a press
    that lost the name and raised before ``start()`` would leave that blocker
    for the life of the process, refusing every later terminate_all() over a
    thread that can never enter PortAudio (#72).

    The worker's ``start()`` can fail -- RuntimeError under a thread-limit
    storm, the same failure ``WarmArm._arm_locked`` handles for captures.
    The worker ledger counts a never-started thread as a blocker (the
    arming window between ``note_worker`` and ``start()`` is exactly what
    it exists to cover), so the failure path must take it back out -- or
    every later terminate_all() is refused for the life of the process by
    a thread that can never enter PortAudio. The recorder-side view rolls
    back with it: recording/recording_thread must not name a launch that
    never happened (stop_recording() would JOIN an unstarted thread, which
    raises). Then the error surfaces to the press caller as it always has.
    """
    if warm is not None:
        # Keep-warm (issue #66): no open, no worker; route the ring.
        warm.begin(sink, lambda: recorder._announce_capture(my_gen))
        return
    if worker is None:
        raise RuntimeError("worker press admitted with no worker")
    try:
        worker.start()
    except BaseException:
        recorder._session.forget_worker(worker)
        with recorder._lock:
            if recorder.recording_thread is worker:
                recorder.recording = False
                recorder.recording_thread = None
        raise


def worker_may_be_native(
    blocked_workers: Callable[[], Sequence[threading.Thread | None]] | None,
) -> bool:
    """True when a named worker may still be inside a native PortAudio
    call (issue #72).

    The names are resolved through the callable AT GATE TIME, not
    captured before the caller's long join: a finished W1 handed in
    while a live W2 is the recorder's current worker must still read
    busy. The thread running the reset is never its own blocker.
    """
    if blocked_workers is None:
        return False
    me = threading.current_thread()
    return any(
        t is not None and t is not me and t.is_alive() for t in blocked_workers()
    )


def warm_press_gate(
    lock: threading.Lock,
    session: PyAudioRegistry,
    press_state: Callable[[], tuple[bool, WarmCapture | None, threading.Thread | None]],
    find_default_input_device: Callable[[pyaudio.PyAudio], int],
    rearm_warm: Callable[[bool], None],
) -> bool:
    """The keep-warm press gate on a lock-consistent snapshot (issue #72): the three fields the gate decides on
    (recording, capture, worker) are read under the recorder lock -- a
    worker may rewrite them between three lock-free reads -- and the
    live-default reset itself runs outside it (native calls, issue #60).
    """
    with lock:
        recording, warm, worker = press_state()
    return session.press_check_if_admissible(
        recording, warm, worker, find_default_input_device, rearm_warm
    )


def current_worker(recorder: AudioRecorder) -> threading.Thread | None:
    """The worker named at CALL time, read under _lock: a name captured
    before a long join is stale by gate time -- a finished W1 must not mask a
    live W2 (issue #72).
    """
    with recorder._lock:
        return recorder.recording_thread


def refresh_recorder_session(
    recorder: AudioRecorder, blocked: threading.Thread | None = None
) -> None:
    """Live-default device reset (issue #72) at the top of the worker.

    Native calls run OUTSIDE _lock (issue #60): they can block and
    stop_recording() needs the lock. One worker runs this at a time;
    ``blocked`` is the worker just joined, and the gate also consults the
    recorder's CURRENT worker at gate time -- which is why the
    name travels as a callable rather than as the thread.
    """
    pa, device_index = recorder._session.snapshot()
    pa, device_index = refresh_pyaudio_session(
        pa,
        device_index,
        find_default_input_device,
        lambda: (blocked, current_worker(recorder)),
    )
    recorder.publish_session(pa, device_index)


def rearm_warm(recorder: AudioRecorder, usable: bool) -> None:
    """Re-arm the keep-warm capture after a reset replaced its session.

    usable=False is the busy path: a capture that would not stop is dead for
    presses (its stop event is set, so it can never re-open) -- drop it and
    later presses take the worker path, not a dead sink. The session keeps the
    handle on its live thread: see WarmKeep.drop().

    usable=True goes through WarmArm.rearm, NOT set_keep_warm: the reset ran
    outside the lock long enough for the menu to be unchecked, and re-arming
    anyway leaves the mic open behind an unchecked item.

    A drop() that reports False names a thread still parked in a native read
    on the session being replaced: arming a fresh capture over it puts two
    readers on PortAudio at once, so the arm is skipped and the recovery flag
    (which drop() set) brings keep-warm back once the ledger clears.
    """
    dead = recorder._warm_arm.take()
    if not recorder._session.warm.drop(dead) or not usable:
        return
    recorder._warm_arm.rearm(recorder._make_warm_capture)


def open_warm_capture_stream(
    recorder: AudioRecorder, pa_module: ModuleType
) -> tuple[pyaudio.Stream, int]:
    """One open for WarmCapture's thread (moved from the recorder, which
    is at its 500-line cap): never overlaps a per-press worker's native
    open on the shared PyAudio (issue #16). ``previous`` is named again
    by the gate at call time -- see ``worker_may_be_native``.
    """
    previous = recorder.recording_thread
    if previous is not None and previous.is_alive():
        previous.join(timeout=PREVIOUS_WORKER_WAIT_SECONDS)
    refresh_recorder_session(recorder, previous)
    # One snapshot for the pair: the index is only a fact about the instance
    # that enumerated it (#72).
    pa, device_index = recorder._session.snapshot()
    new_pa, stream, rate = open_warm_stream(
        pa_module, pa, device_index, recorder.format, recorder.channels,
        recorder.sample_rate, recorder.chunk_size, recorder._session,
    )  # fmt: skip
    if new_pa is not pa:
        with recorder._lock:
            recorder.pyaudio = new_pa
    return stream, rate


def finish_recording(
    recorder: AudioRecorder,
    thread: threading.Thread | None,
    my_gen: int,
    start_monotonic: float | None,
    warm: WarmCapture | None,
    capture_rate: int | None,
):
    """The part of ``stop_recording()`` that runs after the release.

    Joined here rather than kept inline because the recorder is at its
    500-line cap, not because the two halves are unrelated: this is the
    worker join plus the drain, and the ``[audio.stop]`` reason it logs is
    read off observable state (issue #40).
    """
    stuck_open = False
    if thread is not None:
        thread.join(timeout=recorder._stuck_open_timeout_seconds)
        if thread.is_alive():
            # Stuck-open detection: a stuck (not failed) open surfaces as "no
            # speech" because last_error hasn't been written yet -- the join
            # timeout is the only observable signal.
            with recorder._lock:
                if recorder._generation == my_gen:
                    recorder.last_error = "microphone busy — recording did not start"
            print("Recording worker still alive after stop; microphone may be stuck")
            stuck_open = True

    if warm is not None:
        warm.end()
    audio_chunks = []
    while not recorder.audio_queue.empty():
        audio_chunks.append(recorder.audio_queue.get())

    if audio_chunks:
        return assemble_audio(audio_chunks, capture_rate)

    reason = "stuck-open" if stuck_open else "race-lost"
    if warm is not None:
        # The warm stream was not delivering yet: report it (#66).
        with recorder._lock:
            if recorder._generation == my_gen:
                recorder.last_error = "microphone busy — recording did not start"
        reason = "warm-not-ready"
    log_stop(0, start_monotonic, reason)
    return np.array([], dtype=np.float32)


def cleanup_recorder_session(recorder: AudioRecorder) -> None:
    """Clean up PyAudio resources (moved from the recorder, which is at
    its 500-line cap). terminate() is skipped -- and logged -- if a
    worker may still be alive, since terminate() while another thread
    holds a stream is unsafe (lock-scoped local-capture pattern,
    issue #37 task-c).
    """
    if recorder.recording:
        recorder.stop_recording()

    warm = recorder._warm_arm.take()
    if not recorder._session.warm.drop(warm) or not recorder._session.warm.clear():
        print("Warm capture still active at cleanup; skipping PyAudio.terminate()")
        return

    worker_thread = recorder.recording_thread
    with recorder._lock:
        if recorder.recording or worker_alive(worker_thread):
            print(
                "Recording worker still active at cleanup; skipping PyAudio.terminate()"
            )
            return
    recorder.publish_session(None, None)

    # Issue #72: every registered instance goes -- a single instance
    # anywhere keeps PortAudio's init count above zero and frozen.
    if not recorder._session.terminate_all():
        # The registry says why (a blocker it named, or an instance that
        # would not terminate); this line says WHEN, so a log of a quit that
        # left PortAudio initialised can be traced to the quit.
        print("PyAudio instances remained at cleanup; PortAudio may still be up.")


def announce_capture(
    lock: threading.Lock,
    is_announced: Callable[[], bool],
    is_stale: Callable[[], bool],
    mark_announced: Callable[[], None],
    callback: Callable[[], None] | None,
) -> None:
    """Fire on_capture_started at most once per recording cycle (issue #43): the flag check-and-set is one atomic step under the caller's
    lock; the callback runs after the lock is released so it never
    blocks the capture, and a raising callback never kills the thread.
    """
    with lock:
        if is_announced() or is_stale():
            return
        mark_announced()
    if callback is not None:
        try:
            callback()
        except Exception:  # noqa: BLE001 - callback boundary
            # A raising callback must not stop the capture thread.
            print("on_capture_started callback raised")
