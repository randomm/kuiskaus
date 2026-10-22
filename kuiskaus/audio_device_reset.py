"""The device-move reset: what ``refresh()`` actually rebuilds (issue #72).

Split from ``kuiskaus.audio_device`` for the 500-line module cap. The
registry owns the instances and answers "what may I terminate"; this module
owns the reset that consumes that answer -- read the live default, decide
whether it moved, gate on the ledgers, terminate, rebuild, and hand the
keep-warm capture back.

Kept separate because the two questions have different failure modes: a
registry bug leaves an instance unreachable (PortAudio never re-enumerates),
while a reset bug terminates underneath a live thread or replays a frozen
device snapshot. ``PyAudioRegistry.refresh`` is the thin delegation into
here; the behaviour lives beside the reason it exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from kuiskaus.audio_coreaudio import read_live_default_input, session_needs_reset
from kuiskaus.audio_warm_arm import worker_may_be_native
from kuiskaus.audio_worker_keep import terminate_blocker

if TYPE_CHECKING:
    import pyaudio


if TYPE_CHECKING:
    import threading

    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry
    from kuiskaus.audio_warm import WarmCapture


class SessionRefresh(NamedTuple):
    """What the caller writes back under its lock: the session to use and
    the device index resolved against it. ``busy`` means something may still
    be inside a native call, so PortAudio was NOT terminated and the session
    stays dirty for the next press to retry."""

    pyaudio: pyaudio.PyAudio | None
    input_device_index: int | None
    busy: bool


def reset_session(
    registry: PyAudioRegistry,
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
    the dropped capture stays named in the registry's ledger, which
    holds every later terminate back until its thread has left the
    native read (issue #72 review: dropping it from the recorder alone
    orphaned the thread and let the next press terminate underneath it).
    """
    live = read_live_default_input()
    if live is None or not session_needs_reset(live, registry.device_id):
        # ``live is None`` is CoreAudio saying "no evidence" -- never
        # treated as a change, and never as an absence either.
        return SessionRefresh(*registry.snapshot(), False)
    if (
        registry.device_id is not None
        and live.device_id == registry.device_id
        and live == registry.last_reset
    ):
        # A fully successful reset already ran against this exact live
        # read, and the session is actually bound to that device: the
        # live.device_id conjunct (issue #72 review, CRITICAL) makes the
        # guard compare the pair against the session it guards, not just
        # against itself -- device_id and last_reset are now written from
        # one read (registry.open, or the fallback below), and a guard
        # that only compared the pair let split reads pin it to a device
        # the session never opened on, suppressing the still-needed reset
        # forever. The reset exists to stop the "doubt falls to
        # not-alive" fail-safe paying a full terminate-all +
        # Pa_Initialize + re-enumeration on EVERY press forever -- on the
        # keep-warm path, synchronously in the hotkey press callback
        # (issue #72 review). A moved or revived default is a different
        # pair and resets; failed resets clear the pair (open() writes
        # None for both on a construction failure), so they retry.
        return SessionRefresh(*registry.snapshot(), False)

    # Never drive PortAudio's init count to zero while another thread
    # may be inside a native open() or stream.read() on a registered
    # instance: that is the state cleanup() refuses to terminate in
    # (issue #37 task-c), and the pre-#72 poll never reached it because
    # the reset never ran. The ledger (registry.workers / registry.warm)
    # is the authority -- it names every worker this session started,
    # including an issue #60 orphan whose name its successor already
    # cleared, and the capture installed in the arm whether or not this
    # caller handed one over. blocked_workers is the belt to that
    # braces: the threads the caller joined and can speak for, resolved
    # through the callable at GATE time so a finished W1 cannot mask a
    # live W2 (issue #72 review, finding 2). The thread running this
    # reset is never its own blocker. Busy, not half-reset -- nothing
    # here is torn down, so the session stays dirty and the next press
    # asks again.
    blocked = terminate_blocker(registry.workers, registry.warm, exempt=warm)
    if blocked is None and worker_may_be_native(blocked_workers):
        blocked = "named recording worker"
    if blocked is not None:
        return SessionRefresh(*registry.snapshot(), True)

    warm_stopped = registry.warm.drop(warm)
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
        return SessionRefresh(*registry.snapshot(), True)

    if not registry.terminate_all():
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
        registry.warm.need_recovery()
        return SessionRefresh(*registry.snapshot(), True)
    registry.pyaudio = None
    registry.forget_index()
    rebuilt = registry.open(find_default_input_device)
    if not rebuilt:
        # The engine never came back. Leave the session unbound -- the
        # next press re-reads the live device and retries, with
        # nothing left holding PortAudio open.
        #
        # And nothing was re-armed over that missing session: the re-arm
        # belongs after a successful rebuild (issue #72 review, minor).
        # The capture stopped cleanly above, so drop() set no recovery
        # flag -- mirror the terminate_all() bail: hand the dead capture
        # back (a stopped capture is dead for presses) and ask for
        # recovery, so the worker path's recover() brings keep-warm back
        # when the engine does.
        if warm_stopped and warm is not None and rearm_warm is not None:
            rearm_warm(False)
            registry.warm.need_recovery()
        return SessionRefresh(None, None, False)
    if warm_stopped and warm is not None and rearm_warm is not None:
        # Only now does the session exist to arm against.
        rearm_warm(True)
    if registry.last_reset is None:
        # open()'s own read gave no evidence (transient -- quite possible
        # right after the coreaudiod storm that moved the default). Fill
        # BOTH halves of the dedupe baseline from THIS read (issue #72
        # review, round 1 + CRITICAL): the guard needs a non-None
        # device_id to engage, and whichever read it falls back to, the
        # pair it compares has to be the one read that set it. With
        # open()'s read present, open() already wrote both from it.
        registry.last_reset = live
        registry.device_id = live.device_id
    return SessionRefresh(*registry.snapshot(), False)


def press_is_admissible(
    registry: PyAudioRegistry,
    recording: bool,
    warm: WarmCapture | None,
    worker: threading.Thread | None,
    find_default_input_device: Callable[[pyaudio.PyAudio], int],
    rearm_warm: Callable[[bool], None],
) -> bool:
    """The keep-warm press gate (issue #72): run the live-default
    check, or say why this press must not run it.

    The gate lives next to the reset it protects: a press the recorder is
    about to refuse (``recording``) must NOT reach the reset -- it would stop
    the warm capture feeding that live recording and rebuild the session
    underneath it -- and a press with no warm capture has no press-side reset
    to run at all (its worker resets at the top of its own loop). Both are
    admitted (True). An admitted keep-warm press runs the reset itself;
    False = fail this press "microphone busy".
    """
    if recording or warm is None:
        return True
    return not reset_session(
        registry, find_default_input_device, warm, rearm_warm, lambda: (worker,)
    ).busy


# Re-exported for callers that name the entry point rather than the module.
__all__ = ["SessionRefresh", "press_is_admissible", "reset_session"]
