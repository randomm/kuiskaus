"""The input device the recorder's session is bound to (issue #72).

Two halves of one question -- which device am I on, and what has to be
rebuilt when it moves:

* ``read_live_default_input`` asks CoreAudio what the system default
  input is RIGHT NOW. PortAudio snapshots the hardware once at
  ``Pa_Initialize`` and never refreshes it, so a default switch
  mid-session leaves a cached ``PyAudio`` listing a device the system no
  longer uses -- the stale "Built-in Microphone" behind issue #72.
  One-shot ``AudioObjectGetPropertyData`` reads only: no property
  listeners and no runloop, because on this machine
  ``AudioObjectAddPropertyListener`` blocks forever, so a listener-based
  design would reintroduce the issue #60 hang with an unbounded thread.
* ``PyAudioRegistry`` owns every ``PyAudio`` instance the recorder ever
  built, and ``refresh_session`` rebuilds them. PortAudio's teardown is
  reference-counted: one unregistered or unterminated instance keeps
  ``Pa_Terminate`` from releasing the engine, so a "reset" that
  terminates only the instance it can see replays the same frozen
  snapshot. The count reaching zero is the only definition of clean.

The CoreAudio constants are four-char codes and their names are the only
documentation of what a bare integer means, so they are named here.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

import pyaudio

from kuiskaus.audio_retry import (
    close_stream_quietly,
    log_retry_attempt,
    stamp_owner,
    terminate_quietly,
    worker_alive,
)

if TYPE_CHECKING:
    from kuiskaus.audio_warm import WarmCapture

kAudioObjectSystemObject = 1
# The codes below are the PyObjC bridge's own values -- verified against
# kAudioHardwarePropertyDefaultInputDevice / kAudioDevicePropertyDeviceIsAlive
# on a real CoreAudio install. The four-char decodings are 'dIn ' and 'livn'
# (they are NOT 'defi'/'aliv'); the tests mirror the same integers, so a
# comment-following "fix" that retypes one of these numbers breaks live
# detection while the suite stays green.
kAudioHardwarePropertyDefaultInputDevice = 1682533920  # 'dIn '
kAudioDevicePropertyDeviceIsAlive = 1818850926  # 'livn'
kAudioObjectPropertyScopeGlobal = 1735159650  # 'glob'
kAudioObjectPropertyElementMain = 0
kAudioObjectUnknown = 0

# AudioObjectGetPropertyData's last argument is an in/out buffer whose
# address PyObjC must pin; passing None (rather than a preallocated
# bytearray) makes the bridge allocate it and hand the bytes back.
_OUT_BUFFER = None
_SIZE_BYTES = 4
_NO_QUALIFIER = b""


class LiveDefaultInput(NamedTuple):
    """The system default input device as CoreAudio sees it right now."""

    device_id: int
    is_alive: bool


def _read_uint32(object_id: int, selector: int, scope: int, element: int) -> int | None:
    """One UInt32 property read, or None if CoreAudio refused it.

    CoreAudio reports failure as a nonzero OSStatus rather than raising,
    so a None here means "no evidence" -- never "absent". But the PyObjC
    bridge around those calls CAN raise (marshalling under a coreaudiod
    storm, a first-call import failure), and this read runs on the
    AudioRecorder.__init__ and press paths, where an escaping exception
    is an app that never launches. Both failure shapes collapse to None."""
    try:
        import CoreAudio

        address = CoreAudio.AudioObjectPropertyAddress(selector, scope, element)
        status, _size, raw = CoreAudio.AudioObjectGetPropertyData(
            object_id, address, 0, _NO_QUALIFIER, _SIZE_BYTES, _OUT_BUFFER
        )  # fmt: skip
    except Exception as e:  # noqa: BLE001 - PyObjC/CoreAudio boundary, logged
        print(f"CoreAudio property read failed ({e}); treating as no evidence.")
        return None
    if status != 0 or not isinstance(raw, (bytes, bytearray)):
        return None
    return int.from_bytes(raw, sys.byteorder)


def read_live_default_input() -> LiveDefaultInput | None:
    """The live default input device, or None when CoreAudio gave no
    evidence at all (transient failure: a caller must NOT treat that as
    "unchanged" and skip a needed rebuild -- but neither is it proof of
    change, so session_needs_reset() stays quiet on it)."""
    device_id = _read_uint32(
        kAudioObjectSystemObject,
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    if device_id is None or device_id == kAudioObjectUnknown:
        # kAudioObjectUnknown is CoreAudio's "there is no default input
        # device", not a device: as a live read it would make every press
        # rebuild the session on a machine with no microphone, with nothing
        # ever settling. Absence of a device is reported like the absence of
        # an answer -- no evidence, no reset.
        return None
    alive = _read_uint32(
        device_id,
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    # Doubt about liveness falls to "not alive": the safe response to an
    # unproven device is a fresh session, not a stream opened on it.
    return LiveDefaultInput(device_id=device_id, is_alive=alive == 1)


def session_needs_reset(
    live: LiveDefaultInput | None, session_device_id: int | None
) -> bool:
    """Whether the cached session was built against another device.

    A session with no recorded baseline (``session_device_id`` None) has
    never been bound, so any evidence of a live device resets it; no
    evidence at all (``live`` None) resets nothing -- the next press asks
    again."""
    if live is None:
        return False
    return (
        session_device_id is None
        or live.device_id != session_device_id
        or not live.is_alive
    )


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
    worker joins its predecessor before touching anything) and every
    field here is written under the recorder's ``_lock``, the same
    arbiter as ``_capture_rate`` / ``_stream`` (issue #55).
    """

    def __init__(self, pa_module: ModuleType) -> None:
        self._pa_module = pa_module
        self.pyaudio: pyaudio.PyAudio | None = None
        self.input_device_index: int | None = None
        self.device_id: int | None = None
        self._instances: list[pyaudio.PyAudio] = []

    def register(self, pa: pyaudio.PyAudio) -> pyaudio.PyAudio:
        """Track an instance built by this session (the retry loop opens
        its stream on the live one; a warm-up may construct its own) and
        stamp it with its owner, so a caller holding only the instance can
        still find the registry that must terminate it (issue #72)."""
        if pa not in self._instances:
            self._instances.append(pa)
        stamp_owner(pa, self)
        return pa

    def forget(self, pa: pyaudio.PyAudio) -> None:
        """Stop tracking an instance (it stays live -- the caller keeps
        using it, e.g. the instance a new stream is opened on)."""
        if pa in self._instances:
            self._instances.remove(pa)

    def terminate(self, pa: pyaudio.PyAudio) -> None:
        """Terminate and unregister one instance (its stream must be
        closed first: closing the last stream is what releases the
        per-stream init reference)."""
        terminate_quietly(pa)
        self.forget(pa)

    def terminate_all(self) -> None:
        """Terminate every instance this session ever built.

        PortAudio's init count is reference-counted, so a single missed
        instance -- an unregistered one, or one whose terminate() raised
        -- keeps the engine up and the next construction reuses the
        frozen device list. terminate_quietly() swallows per-instance
        failures so one broken instance cannot strand the rest.
        """
        for pa in list(self._instances):
            terminate_quietly(pa)
        self._instances.clear()

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
        try:
            self.input_device_index = find_default_input_device(pa)
        except (OSError, RuntimeError):
            # Transient (coreaudiod storm at startup) or no device at all:
            # the retry loop re-resolves, exactly as before.
            print(
                "Warning: microphone probe failed at startup; "
                "will re-resolve at first recording."
            )
        return True

    def bind_index(self, device_index: int) -> None:
        """Publish the device index a retry attempt resolved (issue #72).

        The loop resolves a device only while the session is unbound; with
        no way to hand the value back (the open-attempt tuple carries the
        rate, not the index) it died with the attempt and every later press
        paid a fresh enumeration. A single-word write, not the composite
        state the recorder's _lock guards: a reader may observe it one
        press late, never torn.
        """
        if self.input_device_index is None:
            self.input_device_index = device_index

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
        index was bound by the attempt itself, via bind_index).
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
        if orphan is not None:
            terminate_quietly(orphan)
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
        blocked_worker: threading.Thread | None = None,
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
        stays dirty for the next press rather than being half-reset.
        """
        live = read_live_default_input()
        if not session_needs_reset(live, self.device_id):
            return SessionRefresh(self.pyaudio, self.input_device_index, False)

        # Never drive PortAudio's init count to zero while another thread
        # may be inside a native open() or stream.read() on a registered
        # instance: that is the state cleanup() refuses to terminate in
        # (issue #37 task-c), and the pre-#72 poll never reached it because
        # the reset never ran. The worker and the warm reopen each name the
        # thread they joined and may still be alive -- issue #60's orphan --
        # and a keep-warm press names the worker it can never speak for.
        # The thread running this reset is never the blocker: it joined its
        # predecessor first. Busy, not half-reset -- nothing here is torn
        # down, so the session stays dirty and the next press asks again.
        if (
            worker_alive(blocked_worker)
            and blocked_worker is not threading.current_thread()
        ):
            return SessionRefresh(self.pyaudio, self.input_device_index, True)

        warm_stopped = warm is None or warm.stop()
        if not warm_stopped:
            # The stuck read owns a capture whose stop event is already
            # set: when the read returns the thread exits and THIS capture
            # can never re-open. Left installed, the next press that needs
            # no reset would be admitted and record silently into a dead
            # sink, so rearm_warm(False) tells the recorder to drop it --
            # later presses take the worker path (issue #72 review).
            print("Keep-warm capture would not stop; dropping it.")
            if rearm_warm is not None:
                rearm_warm(False)
            return SessionRefresh(self.pyaudio, self.input_device_index, True)

        self.terminate_all()
        self.pyaudio = None
        self.input_device_index = None
        rebuilt = self.open(find_default_input_device)
        if warm_stopped and warm is not None and rearm_warm is not None:
            rearm_warm(True)
        if not rebuilt:
            # The engine never came back. Leave the session unbound -- the
            # next press re-reads the live device and retries, with
            # nothing left holding PortAudio open.
            return SessionRefresh(None, None, False)
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
            find_default_input_device, warm, rearm_warm, worker
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
