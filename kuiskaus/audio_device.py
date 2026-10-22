"""The session's PyAudio registry and its device-reset (issue #72).

The "what has to be rebuilt when the device moves" half of the input
device question; ``kuiskaus.audio_coreaudio`` answers "which device am I
on right now" (one-shot CoreAudio reads, no listeners, no runloop) and
its ``session_needs_reset`` decides when that answer has changed.

``PyAudioRegistry`` owns every ``PyAudio`` instance the recorder ever
built, and ``refresh()`` rebuilds them. PortAudio's teardown is
reference-counted: one unregistered or unterminated instance keeps
``Pa_Terminate`` from releasing the engine, so a "reset" that terminates
only the instance it can see replays the same frozen snapshot. The count
reaching zero is the only definition of clean.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import pyaudio

from kuiskaus.audio_coreaudio import LiveDefaultInput, read_live_default_input
from kuiskaus.audio_device_index import DeviceIndexBinding
from kuiskaus.audio_device_reset import (
    SessionRefresh,
    note_terminate_outcome,
    press_is_admissible,
    press_refusal_reason,
    reset_session,
    unavailable_reason,
)
from kuiskaus.audio_rebuild_budget import RebuildBudget
from kuiskaus.audio_retry import (
    close_stream_quietly,
    log_retry_attempt,
    terminate_quietly,
)
from kuiskaus.audio_warm_keep import WarmKeep
from kuiskaus.audio_worker_keep import WorkerKeep, terminate_blocker

if TYPE_CHECKING:
    from kuiskaus.audio_warm import WarmCapture


class PyAudioRegistry:
    """Every PyAudio instance built for one recorder session, plus the
    CoreAudio device they were opened against.

    A plain list, never a WeakSet: an instance PortAudio still counts
    must not be collectable while terminate_all() is trying to reach it,
    and PyAudio exposes no weak references. ``terminate_all()`` is the
    operation this exists for -- it drives PortAudio's init count to zero
    so the next construction re-enumerates the hardware.

    Threading: one thread runs the session's lifecycle at a time (a new
    worker joins its predecessor before touching anything), but the lock
    discipline is narrower than "everything under ``_lock``":
    ``open()``, ``bind_index()`` and ``refresh()`` deliberately run their
    native calls OUTSIDE the recorder's ``_lock`` (they can block for
    seconds and stop_recording() needs that lock, issue #60), and so
    write ``pyaudio`` / ``input_device_index`` / ``device_id`` there.
    What ``_lock`` does arbitrate is the composite state ``adopt()``
    publishes (``_capture_rate`` / ``_stream``, issue #55); the registry
    fields above are single-word writes from whichever thread owns the
    lifecycle, and the instance list is mutated under its own small
    ``_instances_lock`` so a register can never double-append.
    """

    def __init__(self, pa_module: ModuleType) -> None:
        self._pa_module = pa_module
        self.pyaudio: pyaudio.PyAudio | None = None
        # A PortAudio index is a fact about ONE instance's frozen
        # enumeration, so an index whose instance is gone is not an index at
        # all -- reading it back opens a stream on a device that instance
        # never had. It lives in ``_indices``, paired with its instance, and
        # is read through the property below, never through a copy.
        self._indices = DeviceIndexBinding()
        # The (device_id, is_alive) pair the CURRENT session was opened
        # against: a default that stays reported not-alive must cost one
        # rebuild, not one per press -- see the dedupe check in the reset.
        # It is ONE fact from ONE read (open(), or the reset's fallback when
        # open()'s read gave no evidence); ``device_id`` below is derived
        # from it rather than stored beside it, because a baseline split
        # across two fields written by two CoreAudio reads is what let the
        # guard suppress a still-needed reset forever. A failed open clears
        # it -- no instance, no baseline, nothing to dedupe against -- so
        # failed resets stay retryable.
        self.last_reset: LiveDefaultInput | None = None
        # The keep-warm request, plus any capture this session dropped while
        # its thread was parked in a native read: the recorder clears its own
        # ``_warm`` at that same moment, so this is the only handle left that
        # can name those threads (see WarmKeep).
        self.warm = WarmKeep()
        # Every worker this session started. The reset gate used to decide
        # from the two thread names its caller happened to pass, and the
        # recorder drops a wedged worker's name as soon as its successor
        # clears recording_thread -- so the gate could be told "nothing is
        # alive" by a caller that had simply lost the name.
        self.workers = WorkerKeep()
        # Consecutive teardowns an instance refused, and the verdict they add
        # up to. terminate_all() is the one call that knows whether a refusal
        # came from an instance that would not die (counts) or from the ledger
        # gate excusing a thread that may still be native (does not: that is a
        # deferral which releases itself, and blaming it on a dead microphone
        # would send the user to quit the app over a read about to return).
        self.terminate_failures = 0
        self.needs_restart = False
        # Rebuilds since the session last agreed with the live device --
        # the budget that stops a flapping default costing a full teardown
        # on every press. The counters, and why one count is not enough, are
        # on RebuildBudget (kuiskaus.audio_rebuild_budget).
        self.rebuild = RebuildBudget()
        # Monotonic instant the next press may retry a teardown, once
        # ``needs_restart`` has been declared (see ``RESTART_RETRY_SECONDS``).
        self.restart_retry_after = 0.0
        self._instances: list[pyaudio.PyAudio] = []
        # Guards only the _instances list mutations -- never held across a
        # native call, so it cannot be the lock a blocking terminate waits on.
        self._instances_lock = threading.Lock()

    def register(self, pa: pyaudio.PyAudio) -> pyaudio.PyAudio:
        """Track an instance built by this session (the retry loop opens
        its stream on the live one; a warm-up may construct its own). An
        instance PortAudio still counts must be findable by name here --
        ``terminate_all()`` is the only thing that can reach it."""
        with self._instances_lock:
            if pa not in self._instances:
                self._instances.append(pa)
        return pa

    def forget(self, pa: pyaudio.PyAudio) -> None:
        """Stop tracking an instance (it stays live -- the caller keeps
        using it, e.g. the instance a new stream is opened on)."""
        with self._instances_lock:
            if pa in self._instances:
                self._instances.remove(pa)

    def note_worker(self, worker: threading.Thread) -> None:
        """Record a worker the caller is about to start.

        The recorder calls this before ``thread.start()``, so from the instant
        a worker exists the session -- not the caller's memory -- knows it may
        enter a native call. That is what makes "no worker is in PortAudio" a
        question the registry answers for itself, including for the orphan a
        successor already displaced from ``recording_thread`` (issue #60),
        whose name nobody holds any more.
        """
        self.workers.note(worker)

    def forget_worker(self, worker: threading.Thread) -> None:
        """Remove a worker that can never run (its ``start()`` raised).

        A thread that never started counts as a blocker -- the arming window
        is exactly what the ledger exists to cover -- so a launch that failed
        must be taken back out by hand
        or it would refuse every later terminate_all() for the life of the
        process. See ``audio_warm_arm.start_press``, the only caller.
        """
        self.workers.forget(worker)

    @property
    def device_id(self) -> int | None:
        """The device the live session was opened against, or None.

        Read-only, derived from ``last_reset``: the id and the liveness the
        reset dedupes on are one CoreAudio read, and two fields would be two
        chances to disagree.
        """
        return None if self.last_reset is None else self.last_reset.device_id

    @property
    def input_device_index(self) -> int | None:
        """The session's device index while it describes the live instance
        (see ``DeviceIndexBinding``).

        None means "unresolved" -- the answer a caller wants when the binding
        came from an instance that is gone, because re-resolving is the only
        correct response to an index into a device list nobody here has.
        Trusting one was how the mic stayed dead until restart.
        """
        return self._indices.resolve(self.pyaudio)

    def tracks(self, pa: pyaudio.PyAudio) -> bool:
        """Whether this session still tracks ``pa`` as live."""
        with self._instances_lock:
            return pa in self._instances

    def snapshot(self) -> tuple[pyaudio.PyAudio | None, int | None]:
        """The live instance and its resolved index, read as ONE pair.

        An index is only meaningful as a fact about the instance it came from,
        so a caller that wants both takes both from here: two property reads
        leave room for the live instance to be replaced in between, which is
        the mismatch the binding exists to make impossible.
        """
        pa = self.pyaudio
        return pa, self._indices.resolve(pa)

    def set_index(
        self, device_index: int | None, resolved_on: pyaudio.PyAudio | None
    ) -> None:
        """Record the (instance, index) pair the caller just read together.

        The instance is a parameter, not read back from the session: an index
        is only ever a fact about the instance that enumerated it, so the
        caller that holds the pair states whose pair it is. Inferring the
        owner at write time was the bug -- a session that moved between the
        read and this write recorded the index against the wrong instance.
        """
        self._indices.publish(device_index, resolved_on)

    def bind_index(self, device_index: int, resolved_on: pyaudio.PyAudio) -> None:
        """Publish the device index a retry attempt resolved.

        The loop resolves a device only while the session has no usable index;
        with no way to hand the value back (the open-attempt tuple carries the
        rate, not the index) it died with the attempt and every later press
        paid a fresh enumeration. It goes with the instance it was resolved
        against, which is what keeps the hand-back from becoming the wedge: an
        index survived its instance once, and every later attempt then opened
        on a dead device forever.
        """
        self._indices.bind(device_index, resolved_on, self.pyaudio)

    def _dispose(self, pa: pyaudio.PyAudio) -> bool:
        """Terminate one instance and unbind everything that was a fact
        about it. False = it would not terminate, so it stays registered
        (and its index binding stays readable) for the next attempt.
        """
        if not terminate_quietly(pa):
            return False
        self.forget(pa)
        # An index resolved against an instance that has just been terminated
        # describes a device list that no longer exists, so the pairing goes
        # with it: a leftover binding is what let every later press open on a
        # dead index.
        self._indices.unbind(pa)
        return True

    def terminate(self, pa: pyaudio.PyAudio) -> None:
        """Terminate and unregister one instance (its stream must be
        closed first: closing the last stream is what releases the
        per-stream init reference). One that would not terminate stays
        registered: an untracked live instance is exactly what this class
        exists to reach.
        """
        self._dispose(pa)

    def terminate_all(self) -> bool:
        """Terminate every instance this session ever built. True = all gone.

        PortAudio's init count is reference-counted, so a single missed
        instance -- an unregistered one, or one whose terminate() raised --
        keeps the engine up and the next construction reuses the frozen device
        list. terminate_quietly() swallows per-instance failures so one broken
        instance cannot strand the rest, but it reports them: an instance that
        would not terminate STAYS registered, because forgetting a live
        instance is how it becomes permanently unreachable.

        Two different False answers live here, and only one is counted. A
        refusal from the ledger gate -- a worker this session started, or a
        keep-warm capture it installed or dropped, may still be inside a native
        call -- is a deferral that releases itself, so callers treat it as busy
        and nothing is claimed. A refusal from an instance that would not
        terminate is retried and counted, and past ``MAX_TERMINATE_FAILURES``
        in a row the session says so through ``needs_restart`` rather than
        leaving the app reporting Ready over a dead microphone.
        """
        if self._refuse_teardown():
            return False
        incomplete = False
        disposed = 0
        with self._instances_lock:
            pending = list(self._instances)
        # The ledger gains names on other threads (a press notes its worker
        # under the recorder's lock, not this one), so the gate is asked
        # again immediately before anything is disposed: the snapshot above
        # is a list of instances, not a promise about the threads on them.
        if self._refuse_teardown():
            return False
        for pa in pending:
            if self._dispose(pa):
                disposed += 1
            else:
                incomplete = True
        if incomplete and disposed:
            # Say so when the pass was only half a teardown: the session can
            # be left pointing at an instance this call DID terminate (a press
            # then opens on it and gets "microphone unavailable" on healthy
            # hardware until the next press re-binds), and without this line
            # that press has no trace of why.
            print(
                f"Terminated {disposed} of {len(pending)} PyAudio instances; "
                "the session may still name one of them."
            )
        live = self.pyaudio
        if live is not None and not self.tracks(live):
            # The session names an instance this pass terminated. Left bound,
            # the retry loop opens on a corpse, so the name goes with the
            # instance and the next press rebuilds. Unbind rather than clear: a
            # press that rebuilt in the meantime owns the binding now.
            self._indices.unbind(live)
            if self.pyaudio is live:
                self.pyaudio = None
        note_terminate_outcome(self, clean=not incomplete)
        return not incomplete

    def _refuse_teardown(self) -> bool:
        """True (and logged) while something this session started may still
        be inside a native PortAudio call.
        """
        blocker = terminate_blocker(self.workers, self.warm)
        if blocker is None:
            return False
        print(f"{blocker.capitalize()} may still be native; not terminating PyAudio.")
        return True

    def open(self, find_default_input_device: Callable[[pyaudio.PyAudio], int]) -> bool:
        """Construct the persistent cached session (issue #42) and record the
        CoreAudio device it was resolved against -- that baseline is what a
        later live read is compared to. False leaves the session unbound; the
        retry loop still builds a session on demand, and the absent baseline
        makes the next live read re-bind it."""
        try:
            pa = self._pa_module.PyAudio()
        except Exception as construct_error:  # noqa: BLE001 - PyAudio()
            # construction wraps PortAudio's Pa_Initialize(), whose
            # failure modes aren't documented as a narrow exception set.
            print(
                "Warning: PyAudio construction failed "
                f"({str(construct_error)[:120]}); will retry at the next recording."
            )
            # No instance, no baseline: a previous open's pair left standing
            # would have the dedupe guard vouch for a session that is gone.
            self.last_reset = None
            return False
        self.register(pa)
        self.pyaudio = pa
        # ONE read writes the whole dedupe baseline, liveness and device id
        # together: a reset whose read gave no evidence fills it from its own
        # read instead (see audio_device_reset).
        self.last_reset = read_live_default_input()
        # A fresh instance re-enumerates, so whatever was resolved before
        # describes another instance's device list.
        self._indices.clear()
        try:
            self.bind_index(find_default_input_device(pa), pa)
        except (OSError, RuntimeError):
            # Transient (coreaudiod storm) or no device at all: the retry
            # loop re-resolves, exactly as before.
            print(
                "Warning: microphone probe failed; "
                "will re-resolve at the next recording."
            )
        return True

    def adopt(
        self,
        lock: threading.Lock,
        is_superseded: Callable[[], bool],
        publish_capture_rate: Callable[[int | None], None],
        stream_is_open: Callable[[], bool],
        new_pa: pyaudio.PyAudio,
        stream: pyaudio.Stream,
        capture_rate: int | None,
        attempt: int,
        attempt_start: float,
        max_attempts: int,
    ) -> bool:
        """Lock-scoped ownership transfer of a successful open into the
        session. Returns False (stream closed, abort logged) if the retry
        sequence was superseded while it ran.

        Every attempt opens on the single live instance, so ``new_pa`` is
        normally the instance already in the registry; a different one only
        shows up when the session was never bound -- the fresh instance then
        becomes the live one, and the attempt already bound its index paired
        with itself, so the binding reads only while it IS the live one.
        """
        orphan: pyaudio.PyAudio | None = None
        previous: pyaudio.PyAudio | None = None
        superseded = False
        abandoned: pyaudio.Stream | None = None
        with lock:
            if is_superseded():
                # The stream this worker opened and no longer owns. Its close
                # is a native call, so it is only DEFERRED here and run after
                # the release -- never inside the recorder lock, which
                # stop_recording() needs (issue #60).
                abandoned = stream
                # An instance this worker itself constructed (the session never
                # had one): the worker owns its termination, and must actually
                # take it -- attempt_open_once registers what it builds, so an
                # abandoned instance stays registered and holds PortAudio's
                # init count above zero until some later terminate_all() gets
                # to it, which on this path may never come. The recorder's own
                # live session is never touched here. Untracking happens under
                # the lock; terminate() and the stream close run outside it.
                if new_pa is not self.pyaudio:
                    orphan = new_pa
                    self.forget(new_pa)
                superseded = True
            else:
                # Capture the previous instance and publish the new one plus
                # its capture rate under the SAME lock acquisition
                # (issue #55: rate + stream must be atomic for
                # stop_recording()).
                previous = self.pyaudio
                self.register(new_pa)
                self.pyaudio = new_pa
                publish_capture_rate(capture_rate)
        if abandoned is not None:
            # Outside the lock: closing a stream is a native call, and this
            # one is entirely this worker's own (no other thread has ever
            # seen it), so nothing else can be waiting on its completion.
            close_stream_quietly(abandoned)
        if orphan is not None and not self._dispose(orphan):
            # An orphan that would not terminate goes back on the list: it is
            # still live, still holding PortAudio's init count, and the next
            # terminate_all() has to be able to find it.
            self.register(orphan)
        if superseded:
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            return False
        # Outside the lock: whoever captured the displaced instance owns its
        # termination (cleanup() uses the same pattern, so a double terminate
        # is impossible by construction).
        #
        # Invariant: nothing else opens a stream concurrently here -- only
        # _recording_worker does, AFTER the open returns -- so stream_is_open()
        # is a defensive guard, not a racy gate. And the displaced instance is
        # only a DIFFERENT instance after a device reset: the retry loop reuses
        # the single live instance on every attempt, so terminating "the
        # previous" unconditionally would terminate the one this stream is open
        # on.
        if previous is not None and previous is not new_pa and not stream_is_open():
            self.terminate(previous)
        return True

    def refresh(
        self,
        find_default_input_device: Callable[[pyaudio.PyAudio], int],
        warm: WarmCapture | None = None,
        rearm_warm: Callable[[bool], None] | None = None,
        blocked_workers: Callable[[], Sequence[threading.Thread | None]] | None = None,
    ) -> SessionRefresh:
        """Reset the session when the live default moved (issue #72).

        The behaviour lives in ``kuiskaus.audio_device_reset`` (the
        module-size seam); this is the registry's entry point into it. See
        there for when it runs, what it refuses, and what "busy" costs the
        press.
        """
        return reset_session(
            self, find_default_input_device, warm, rearm_warm, blocked_workers
        )

    def press_check_if_admissible(
        self,
        recording: bool,
        warm: WarmCapture | None,
        worker: threading.Thread | None,
        find_default_input_device: Callable[[pyaudio.PyAudio], int],
        rearm_warm: Callable[[bool], None],
    ) -> bool:
        """The keep-warm press gate; the rule itself is
        ``audio_device_reset.press_is_admissible``."""
        return press_is_admissible(
            self, recording, warm, worker, find_default_input_device, rearm_warm
        )

    def press_refusal_reason(self) -> str:
        """What a refused press is called -- "busy" while a retry is still
        possible, the restart verdict once a teardown has spent the budget
        (``audio_device_reset.press_refusal_reason``)."""
        return press_refusal_reason(self)

    def unavailable_reason(self, last_error: BaseException | None) -> str:
        """What an exhausted retry loop is called for this session
        (``audio_device_reset.unavailable_reason``)."""
        return unavailable_reason(self, last_error)

    def forget_index(self) -> None:
        """Drop the device-index binding entirely (the reset does it once
        every instance the index was resolved against is gone)."""
        self._indices.clear()
