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
from typing import TYPE_CHECKING, NamedTuple

import pyaudio

from kuiskaus.audio_coreaudio import (
    LiveDefaultInput,
    read_live_default_input,
    session_needs_reset,
)
from kuiskaus.audio_device_index import DeviceIndexBinding
from kuiskaus.audio_retry import (
    close_stream_quietly,
    log_retry_attempt,
    stamp_owner,
    terminate_quietly,
)
from kuiskaus.audio_warm_arm import worker_may_be_native
from kuiskaus.audio_warm_keep import WarmKeep

if TYPE_CHECKING:
    from kuiskaus.audio_warm import WarmCapture


class SessionRefresh(NamedTuple):
    """What the caller writes back under its lock: the session to use and
    the device index resolved against it. ``busy`` means the keep-warm
    stream would not stop, so PortAudio was NOT terminated and the
    session stays dirty for the next press to retry."""

    pyaudio: pyaudio.PyAudio | None
    input_device_index: int | None
    busy: bool


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
        # The device index lives in ``_indices``, paired with the instance it
        # was resolved against (issue #72 review, round 1): a PortAudio index
        # is a fact about ONE instance's frozen enumeration, so an index
        # whose instance is gone is not an index at all -- reading it back
        # opens a stream on a device that instance never had. Read it
        # through the property below, never through a copy.
        self._indices = DeviceIndexBinding()
        self.device_id: int | None = None
        # The (device_id, is_alive) pair the last FULLY SUCCESSFUL reset
        # was performed against (issue #72 review): a default that stays
        # reported not-alive must cost one rebuild, not one per press --
        # see the dedupe check in refresh(). Only written after open()
        # came back, so failed resets stay retryable.
        self._last_reset: LiveDefaultInput | None = None
        # The keep-warm request, plus any capture this session dropped while
        # its thread was parked in a native read: the recorder clears its own
        # ``_warm`` at that same moment, so this is the only handle left that
        # can name those threads (see WarmKeep).
        self.warm = WarmKeep()
        self._instances: list[pyaudio.PyAudio] = []
        # Guards only the _instances list mutations -- never held across a
        # native call, so it cannot be the lock a blocking terminate waits on.
        self._instances_lock = threading.Lock()

    def register(self, pa: pyaudio.PyAudio) -> pyaudio.PyAudio:
        """Track an instance built by this session (the retry loop opens
        its stream on the live one; a warm-up may construct its own) and
        stamp it with its owner, so a caller holding only the instance can
        still find the registry that must terminate it (issue #72)."""
        if pa not in self._instances:
            with self._instances_lock:
                if pa not in self._instances:
                    self._instances.append(pa)
        stamp_owner(pa, self)
        return pa

    def forget(self, pa: pyaudio.PyAudio) -> None:
        """Stop tracking an instance (it stays live -- the caller keeps
        using it, e.g. the instance a new stream is opened on)."""
        with self._instances_lock:
            if pa in self._instances:
                self._instances.remove(pa)

    @property
    def input_device_index(self) -> int | None:
        """The session's device index, while it describes the live instance
        (issue #72 review, round 1; see ``DeviceIndexBinding``).

        None means "unresolved", and it is the answer a caller wants when
        the binding came from an instance that is gone: re-resolving is the
        only correct response to an index into a device list nobody here
        has, and trusting one was how the mic stayed dead until restart.
        """
        return self._indices.resolve(self.pyaudio)

    def set_index(self, device_index: int | None) -> None:
        """Record a (live instance, index) pair the caller holds together.

        The recorder's session write-back comes through here: it read the
        pair together, so the pairing is recorded as it was read rather than
        inferred afterwards (issue #72 review, round 1).
        """
        self._indices.publish(device_index, self.pyaudio)

    def bind_index(self, device_index: int, resolved_on: pyaudio.PyAudio) -> None:
        """Publish the device index a retry attempt resolved (issue #72).

        The loop resolves a device only while the session has no usable
        index; with no way to hand the value back (the open-attempt tuple
        carries the rate, not the index) it died with the attempt and every
        later press paid a fresh enumeration.

        It is published WITH the instance it was resolved against, which is
        what keeps the hand-back from becoming the wedge: an index survived
        its instance once, and every later attempt then opened on a dead
        device forever (issue #72 review, round 1). A binding already
        serving the live instance stands -- two attempts in one press can
        both resolve, and the one against the instance being opened on is
        the one to keep.
        """
        if self.input_device_index is not None:
            return
        self._indices.bind(device_index, resolved_on)

    def _dispose(self, pa: pyaudio.PyAudio) -> bool:
        """Terminate one instance and unbind everything that was a fact
        about it. False = it would not terminate, so it stays registered
        (and its index binding stays readable) for the next attempt.
        """
        if not terminate_quietly(pa):
            return False
        self.forget(pa)
        # An index resolved against an instance that has just been
        # terminated describes a device list that no longer exists, so the
        # pairing goes with it (issue #72 review, round 1): a leftover
        # binding is what let every later press open on a dead index.
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
        instance -- an unregistered one, or one whose terminate() raised
        -- keeps the engine up and the next construction reuses the
        frozen device list. terminate_quietly() swallows per-instance
        failures so one broken instance cannot strand the rest, but it
        reports them: an instance that would not terminate STAYS
        registered, because forgetting a live instance is how it becomes
        permanently unreachable -- the next attempt gets another shot.

        False also means refused: a keep-warm capture this session dropped
        whose thread may still be reading on one of these instances. Callers
        treat False as busy rather than as a clean session.
        """
        if not self.warm.clear():
            print("Keep-warm capture may still be reading; not terminating PyAudio.")
            return False
        incomplete = False
        with self._instances_lock:
            pending = list(self._instances)
        for pa in pending:
            if not self._dispose(pa):
                incomplete = True
        return not incomplete

    def open(self, find_default_input_device: Callable[[pyaudio.PyAudio], int]) -> bool:
        """Construct the persistent cached session (issue #42) and record
        the CoreAudio device it was resolved against (issue #72) -- that
        baseline is what a later live read is compared to. False leaves the
        session unbound (every field None); the retry loop still builds a
        session on demand, and the absent baseline makes the next live read
        re-bind it."""
        try:
            pa = self._pa_module.PyAudio()
        except Exception as construct_error:  # noqa: BLE001 - PyAudio()
            # construction wraps PortAudio's Pa_Initialize(), whose
            # failure modes aren't documented as a narrow exception set.
            print(
                "Warning: PyAudio construction failed at startup "
                f"({construct_error}); will retry at first recording."
            )
            return False
        self.register(pa)
        self.pyaudio = pa
        live = read_live_default_input()
        self.device_id = None if live is None else live.device_id
        # A fresh instance re-enumerates, so whatever was resolved before
        # describes another instance's device list (issue #72 review, r1).
        self._indices.clear()
        try:
            self.bind_index(find_default_input_device(pa), pa)
        except (OSError, RuntimeError):
            # Transient (coreaudiod storm at startup) or no device at all:
            # the retry loop re-resolves, exactly as before.
            print(
                "Warning: microphone probe failed at startup; "
                "will re-resolve at first recording."
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

        Under issue #72 every attempt opens on the single live instance,
        so ``new_pa`` is normally the instance already in the registry;
        a different one only shows up when the session was never bound --
        the fresh instance then becomes the live one (its resolved device
        index was bound by the attempt itself, via bind_index -- paired with
        that instance, so the binding is readable only while it IS the live
        one; issue #72 review round 1).
        """
        orphan: pyaudio.PyAudio | None = None
        previous: pyaudio.PyAudio | None = None
        superseded = False
        with lock:
            if is_superseded():
                close_stream_quietly(stream)
                # An instance this worker itself constructed (the session
                # never had one): the worker owns its termination. It must
                # actually take it -- attempt_open_once registers what it
                # builds, so an abandoned instance is registered and only
                # terminate_all() can ever see it; while the session stays
                # unbound, registry_for(None) is None, refresh() cannot run,
                # and every superseded press leaves another live instance
                # holding PortAudio's init count above zero: the frozen
                # device list this module exists to clear. The recorder's
                # own live session is never touched here -- a later press
                # still needs it, and the retry loop opens on that instance.
                # Untracking happens under the lock; the terminate() runs
                # outside it. close_stream_quietly() just above is the one
                # native call that still runs under the lock: it closes
                # THIS worker's own abandoned stream (pre-existing shape,
                # never another thread's in-flight I/O).
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
        if orphan is not None and not self._dispose(orphan):
            # An orphan that would not terminate goes back on the list: it is
            # still live, still holding PortAudio's init count, and the next
            # terminate_all() has to be able to find it.
            self.register(orphan)
        if superseded:
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            return False
        # Outside the lock: whoever captured the displaced instance owns
        # its termination (cleanup() uses the same pattern, so a double
        # terminate is impossible by construction).
        #
        # Invariant (lens review HIGH #3): nothing else opens a stream
        # concurrently here -- only _recording_worker does, AFTER the open
        # returns -- so stream_is_open() is a defensive guard, not a racy
        # gate. And the displaced instance is only a DIFFERENT instance
        # after a device reset: the retry loop reuses the single live
        # instance on every attempt (issue #72), so terminating "the
        # previous" unconditionally would terminate the one this stream is
        # open on.
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

        Runs OUTSIDE the recorder lock: these are native calls that can
        block for seconds, and stop_recording() needs that lock. The
        trigger is a LIVE CoreAudio read -- not a PortAudio device-count
        comparison, which is the bug this replaces: PortAudio snapshots
        the hardware at Pa_Initialize, so a count comparison against a
        cached instance can never see the switch.

        With keep-warm on there is no worker, so the press itself runs
        this; the warm stream is stopped BEFORE terminating because
        closing the last open stream is what releases PortAudio's per
        -stream init reference. A warm capture that would not stop means
        the engine is still up, so this returns busy and the session
        stays dirty for the next press rather than being half-reset -- and
        the dropped capture stays named in self.warm, which holds every
        later terminate back until its thread has left the native read
        (issue #72 review: dropping it from the recorder alone orphaned the
        thread and let the next press terminate underneath it).
        """
        live = read_live_default_input()
        if live is None or not session_needs_reset(live, self.device_id):
            # ``live is None`` is CoreAudio saying "no evidence" -- never
            # treated as a change, and never as an absence either.
            return SessionRefresh(self.pyaudio, self.input_device_index, False)
        if self.device_id is not None and live == self._last_reset:
            # A fully successful reset already ran against this exact live
            # read. refresh() re-records device_id but never the liveness it
            # reset against, so the "doubt falls to not-alive" fail-safe
            # would otherwise pay a full terminate-all + Pa_Initialize +
            # re-enumeration on EVERY press forever -- on the keep-warm
            # path, synchronously in the hotkey press callback (issue #72
            # review). A moved or revived default is a different pair and
            # resets; failed resets recorded nothing (``_last_reset`` is
            # written only after a successful open), so they retry.
            return SessionRefresh(self.pyaudio, self.input_device_index, False)

        # Never drive PortAudio's init count to zero while another thread
        # may be inside a native open() or stream.read() on a registered
        # instance: that is the state cleanup() refuses to terminate in
        # (issue #37 task-c), and the pre-#72 poll never reached it because
        # the reset never ran. The worker and the warm reopen each name the
        # thread they joined -- issue #60's orphan -- and a keep-warm press
        # names the worker it can never speak for; the names resolve at
        # GATE time, so a finished W1 cannot mask a live W2 (issue #72
        # review, finding 2). The thread running this reset is never the
        # blocker: it joined its predecessor first. Busy, not half-reset --
        # nothing here is torn down, so the session stays dirty and the
        # next press asks again.
        if worker_may_be_native(blocked_workers) or not self.warm.clear():
            return SessionRefresh(self.pyaudio, self.input_device_index, True)

        warm_stopped = self.warm.drop(warm)
        if not warm_stopped:
            # The stuck read owns a capture whose stop event is already
            # set: when the read returns the thread exits and THIS capture
            # can never re-open. Left installed, the next press that needs
            # no reset would be admitted and record silently into a dead
            # sink, so rearm_warm(False) tells the recorder to drop it --
            # later presses take the worker path (issue #72 review).
            #
            # Dropped for presses, NOT gone for PortAudio: WarmKeep kept
            # the only remaining handle on that thread, and that handle is
            # what gates every later terminate_all() until the thread has
            # actually exited. The hand-back below asks for that SAME object
            # to be dropped again; WarmKeep.drop() answers with a probe, not
            # a second join (#72 review, round 2).
            print("Keep-warm capture would not stop; dropping it.")
            if rearm_warm is not None:
                rearm_warm(False)
            return SessionRefresh(self.pyaudio, self.input_device_index, True)

        if not self.terminate_all():
            # Busy: the session stays dirty and the next press retries.
            # But the warm capture was already stopped cleanly above, and
            # a stopped capture is dead for presses either way -- mirror
            # the sibling branch and hand it back, or the next press that
            # needs no reset (the default settled back, or one transient
            # no-evidence read) is admitted and records the stale pre-roll
            # into a dead sink with no error surfaced (issue #72 review,
            # round 4). And ask for recovery explicitly: stop() SUCCEEDED,
            # so drop() never set _recover_pending and keep-warm would
            # stay off -- menu checkmark and all -- for the life of the
            # process.
            if warm_stopped and warm is not None and rearm_warm is not None:
                rearm_warm(False)
            self.warm.need_recovery()
            return SessionRefresh(self.pyaudio, self.input_device_index, True)
        self.pyaudio = None
        self._indices.clear()
        rebuilt = self.open(find_default_input_device)
        if warm_stopped and warm is not None and rearm_warm is not None:
            rearm_warm(True)
        if not rebuilt:
            # The engine never came back. Leave the session unbound -- the
            # next press re-reads the live device and retries, with
            # nothing left holding PortAudio open.
            return SessionRefresh(None, None, False)
        # Dedupe baseline: this exact live read now has a session built
        # against it (see the dedupe check above). device_id is filled from
        # THIS read when open()'s second read gave no evidence: the dedupe
        # guard needs a non-None device_id to engage, and leaving it None
        # (open()'s transient failure is quite possible right after a
        # coreaudiod storm) meant a full rebuild on every press -- on the
        # hotkey callback thread in the keep-warm path, the exact hazard
        # the dedupe was added to remove (issue #72 review, round 1).
        self._last_reset = live
        if self.device_id is None:
            self.device_id = live.device_id
        return SessionRefresh(self.pyaudio, self.input_device_index, False)

    def press_check_if_admissible(
        self,
        recording: bool,
        warm: WarmCapture | None,
        worker: threading.Thread | None,
        find_default_input_device: Callable[[pyaudio.PyAudio], int],
        rearm_warm: Callable[[bool], None],
    ) -> bool:
        """The keep-warm press gate (issue #72): run the live-default
        check, or say why this press must not run it.

        The gate lives here, next to the reset it protects: a press the
        recorder is about to refuse (``recording``) must NOT reach
        refresh() -- it would stop the warm capture feeding that live
        recording and rebuild the session underneath it -- and a press
        with no warm capture has no press-side reset to run at all (its
        worker refreshes at the top of its own loop). Both are admitted
        (True). An admitted keep-warm press runs the reset itself; False
        = fail this press "microphone busy".
        """
        if recording or warm is None:
            return True
        return not self.refresh(
            find_default_input_device, warm, rearm_warm, lambda: (worker,)
        ).busy


def check_superseded(
    lock: threading.Lock,
    is_superseded: Callable[[], bool],
    attempt: int,
    max_attempts: int,
    attempt_start: float,
) -> bool:
    """True (and logged) when this retry sequence lost its race: a newer
    generation took over, or recording was released. Shared by the
    pre-sleep and post-sleep abort sites (issue #37 lens review MEDIUM
    #7: previously duplicated). The predicate is evaluated under the
    caller's lock, which is the single arbiter of that state (issue #55).
    """
    with lock:
        if is_superseded():
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            return True
        return False
