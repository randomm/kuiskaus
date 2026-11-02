"""The device-move reset: what ``refresh()`` actually rebuilds (issue #72).

Split from ``kuiskaus.audio_device``, which owns the instances themselves. The
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

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from kuiskaus.audio_coreaudio import (
    LiveDefaultInput,
    read_live_default_input,
    session_needs_reset,
)
from kuiskaus.audio_rebuild_budget import MAX_RESET_REBUILDS
from kuiskaus.audio_reset_outcome import SessionRefresh, unchanged
from kuiskaus.audio_worker_keep import terminate_blocker, worker_may_be_native

if TYPE_CHECKING:
    import threading

    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry
    from kuiskaus.audio_warm import WarmCapture

# The one true thing to say once the session has given up on tearing PortAudio
# down. It lives beside the other press-failure text because both are what the
# user reads in the menubar's 🔴 line, and the difference between them is the
# point.
MICROPHONE_NEEDS_RESTART = (
    "Microphone needs an app restart — PortAudio would not release the device"
)

# The other give-up: the teardown succeeded every time and the engine is
# fine, but CoreAudio keeps reporting a different pair for the same device,
# so the session cannot settle on it. Telling this user PortAudio would not
# release the device is a false claim about a teardown that never failed,
# and sending them to quit the app over a device that may settle in seconds
# is the wrong ask -- the backoff retries on its own.
MICROPHONE_NOT_SETTLING = "Microphone is not settling — try again in a moment"

# How many consecutive teardowns an instance may refuse before the session
# stops implying the microphone might come back (issue #72).
# Retrying forever is right -- forgetting a live instance would leave
# PortAudio's init count above zero and make every later rebuild useless --
# but retrying forever *silently* is not: with no verdict the press reports
# busy on every press, device_id stays pinned to a dead device, last_error
# stays None, and the menubar shows Ready over a microphone that will never
# come back. Three is "the storm is not transient", and it still leaves room
# for one.
MAX_TERMINATE_FAILURES = 3

# How long a session that gave up on its teardown waits before asking
# PortAudio again. While ``needs_restart`` is set the teardown is known
# futile, and re-running it on every press costs the hotkey callback that
# full stall for no new information.
RESTART_RETRY_SECONDS = 30.0

# Test seam for the backoff clock.
_now = time.monotonic


def note_terminate_outcome(registry: PyAudioRegistry, clean: bool) -> None:
    """Book one teardown attempt against the budget.

    Only a teardown that actually ran gets here -- a refusal from the ledger
    gate (a thread that may still be native) is a deferral that releases
    itself, and blaming it on a dead microphone would send the user to quit
    the app over a read that was about to return.

    A clean teardown means the instance that would not die is gone, so the
    verdict goes back to nothing and a later storm pays for itself from zero:
    the claim must describe the teardown as it is now, not the worst it ever
    was. The rebuild budget is NOT cleared here -- every reset tears down
    cleanly by construction, and clearing it would make the counters a
    constant zero.
    """
    if clean:
        registry.terminate_failures = 0
        registry.needs_restart = False
        return
    registry.terminate_failures += 1
    if registry.needs_restart:
        # Already given up: re-arm the backoff so the next press does not
        # pay for the same futile teardown, and stop there. What the user is
        # now told follows from ``terminate_failures`` against
        # ``MAX_TERMINATE_FAILURES`` (see ``restart_verdict``): a streak
        # verdict with a first refusal on top still says "not settling" --
        # one refusal has not earned the PortAudio claim -- and the stronger
        # wording replaces it on its own once the refusals reach the budget.
        registry.restart_retry_after = _now() + RESTART_RETRY_SECONDS
        return
    if registry.terminate_failures < MAX_TERMINATE_FAILURES:
        return
    _declare_needs_restart(
        registry,
        f"PyAudio refused to terminate {registry.terminate_failures} times "
        "running; the microphone needs an app restart.",
    )


def restart_verdict(registry: PyAudioRegistry) -> str:
    """The give-up's user-facing claim, read off the state that holds it.

    The two verdicts say different things and only one of them is allowed to
    blame PortAudio: "would not release the device" is a claim about a
    teardown, so it belongs to the session that has ``MAX_TERMINATE_FAILURES``
    consecutive refusals on the record (which a clean teardown clears along
    with the verdict -- see ``note_terminate_outcome``). A single refusal is
    not that claim: the streak verdict is declared with every teardown
    released cleanly, and one later refusal is one data point, not a storm --
    this user's microphone may still come back on the next press, and the
    backoff retries on its own. Choosing the wording here rather than storing
    it is what keeps the sentence true after the state that earned it moves
    -- the same reason the streak itself is not frozen into a message.
    """
    if registry.terminate_failures >= MAX_TERMINATE_FAILURES:
        return MICROPHONE_NEEDS_RESTART
    return MICROPHONE_NOT_SETTLING


def _declare_needs_restart(registry: PyAudioRegistry, reason: str) -> None:
    """Give up on the teardown, say so, and arm the retry backoff."""
    registry.needs_restart = True
    registry.restart_retry_after = _now() + RESTART_RETRY_SECONDS
    print(reason)


def _teardown_backoff_open(registry: PyAudioRegistry) -> bool:
    """True while a declared give-up still excuses the teardown.

    The window is what makes the verdict cheap to hold: the session says no
    to the one call it has just proved futile, and asks again once the
    window has expired.
    """
    return registry.needs_restart and _now() < registry.restart_retry_after


def _press_is_new_evidence(registry: PyAudioRegistry, live: LiveDefaultInput) -> bool:
    """True when this press reports something the verdict on the record does
    not already cover.

    Two verdicts can be on the record and only one of them is about a
    device. The rebuild budget counts devices that will not settle, so a
    press asking about a device the session has never rebuilt against
    during this run is not that repetition -- it is the user moving the
    default by hand, the whole condition issue #72 exists to follow, and it
    is attempted inside the window. A device the budget HAS rebuilt against
    is the counted pair bouncing back: a default alternating A<->B presents
    a "different device" on every press, so comparing against the last one
    counted alone would leave the exemption permanently open and spend the
    full futile teardown on every press -- the stall the window exists to
    prevent.

    The teardown's own verdict is not about a device at all -- PortAudio
    refusing an instance says the ENGINE will not come down -- so a
    different device is no new evidence there and the window still holds.

    A pure read: the exemption itself is claimed by ``reset_session``, at
    the point of no return after every deferral gate (ledger, keep-warm
    stop, teardown) has passed -- a press any of them defers must not pay
    for a rebuild it never ran.
    """
    return (
        not registry.terminate_failures
        and registry.rebuild.device is not None
        and registry.rebuild.is_new_device(live.device_id)
    )


def _hand_back_capture(
    registry: PyAudioRegistry,
    warm: WarmCapture | None,
    rearm_warm: Callable[[bool], None] | None,
    *,
    ask_recovery: bool,
) -> None:
    """Tell the recorder what became of the capture this reset stopped.

    A capture stop() reported stopped is dead for presses either way -- its
    stop event is set, so it can never re-open -- so it is handed back rather
    than being left installed as a sink that silently swallows the next take.
    The hand-back is always the drop (``rearm_warm(False)``): no reset path
    arms a fresh capture from here -- not the bail-outs (no session to arm
    against), and not the completed rebuild either, because a capture armed
    mid-press has not opened its stream yet and routing that press into it
    would discard the take (see ``_rebuild``). Keep-warm comes back through
    the worker's ``WarmKeep.recover()`` instead.

    ``ask_recovery`` is the separate ask for the bail-out paths: stop()
    SUCCEEDED there, so ``drop()`` set no recovery flag and keep-warm would
    stay off -- menu checkmark and all -- for the life of the process.

    ``WarmArm`` re-raises a capture whose ``start()`` failed, having already
    undone the install and filed the recovery ask. Nothing is gained by
    carrying that raise up through the reset into the hotkey callback: the
    press the user just made would die instead of falling back to the worker
    path, and a later press recovers the capture through the flag the arming
    already set.
    """
    if ask_recovery:
        registry.warm.need_recovery()
    if warm is None or rearm_warm is None:
        return
    try:
        rearm_warm(False)
    except Exception as arm_error:  # noqa: BLE001 - press-path boundary, logged
        print(
            f"Keep-warm hand-back failed ({str(arm_error)[:120]}); "
            "the recording goes on without it."
        )


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
    trigger is a LIVE CoreAudio read -- PortAudio snapshots the hardware
    at Pa_Initialize, so a device-count comparison against a cached
    instance can never see the switch.

    With keep-warm on there is no worker, so the press itself runs this;
    the warm stream is stopped BEFORE terminating because closing the last
    open stream is what releases PortAudio's per-stream init reference. A
    capture that would not stop means the engine is still up: this returns
    busy and the session stays dirty for the next press rather than being
    half-reset, and the dropped capture stays named in the registry's
    ledger, which holds every later terminate back until its thread has
    left the native read.

    Serialised (issue #72): the press (via the keep-warm gate), the worker
    and the warm capture all enter, and two can arrive at once. The
    registry's ``_reset_lock`` makes one of them the reset; the others get
    the deferral answer immediately (``locked``) -- never a wait, which
    would park the hotkey callback behind another thread's teardown (#60).
    """
    if not registry._reset_lock.acquire(blocking=False):
        return unchanged(registry, busy=True, locked=True)
    try:
        live = read_live_default_input()
        if live is None:
            # CoreAudio saying "no evidence": never treated as a change, and
            # never as an absence either. Nothing was proved about the session,
            # so nothing here may clear a verdict an earlier press earned.
            return unchanged(registry)
        settled = _settled_or_refused(registry, live)
        if settled is not None:
            return settled
        blocked = _native_blocker(registry, warm, blocked_workers)
        if blocked is not None:
            print(f"Device reset deferred: {blocked} may still be native.")
            return unchanged(registry, busy=True)
        # Captured BEFORE the teardown, and the exemption is claimed after
        # it (see below): a clean terminate clears ``needs_restart``, so a
        # re-read at the claim would always say "closed" at this point.
        backoff = _teardown_backoff_open(registry)
        if backoff and not registry.rebuild.exemption_available():
            # Refused HERE, before anything is torn down: the allowance's
            # whole point is that a fast device-id rotation must not pay the
            # full futile teardown on every press once it is spent.
            return unchanged(registry, busy=True)
        if not registry.warm.drop(warm):
            # The stuck read owns a capture whose stop event is already set:
            # when the read returns the thread exits and THIS capture can never
            # re-open. Left installed, the next press that needs no reset would
            # be admitted and record silently into a dead sink, so the
            # hand-back below tells the recorder to drop it -- later presses
            # take the worker path.
            #
            # Dropped for presses, NOT gone for PortAudio: WarmKeep kept the
            # only remaining handle on that thread, and that handle is what
            # gates every later terminate_all() until the thread has actually
            # exited. The hand-back asks for that SAME object to be dropped
            # again; WarmKeep.drop() answers with a probe, not a second join.
            print("Keep-warm capture would not stop; dropping it.")
            _hand_back_capture(registry, warm, rearm_warm, ask_recovery=False)
            return unchanged(registry, busy=True)
        if not registry.terminate_all():
            # Busy: the session stays dirty and the next press retries. The
            # capture was stopped cleanly above and is dead for presses
            # either way, so hand it back and ask for recovery (see
            # _hand_back_capture).
            _hand_back_capture(registry, warm, rearm_warm, ask_recovery=True)
            return unchanged(registry, busy=True)
        if backoff:
            # The point of no return: the teardown ran clean and the rebuild
            # is next, so THIS press pays for it. Claiming above the two
            # deferral bails let a press that never rebuilt spend the
            # allowance -- the rule the ledger gate already honours.
            registry.rebuild.claim_exemption()
        return _rebuild(registry, live, warm, rearm_warm, find_default_input_device)
    finally:
        registry._reset_lock.release()


def _settled_or_refused(
    registry: PyAudioRegistry, live: LiveDefaultInput
) -> SessionRefresh | None:
    """Whether this press needs a rebuild at all, or None if it does.

    Two refusals, in the order the caller read them: the session already
    agrees with the system (settle the budget and stand down), or the
    teardown backoff is still open on evidence this press has not changed.
    """
    if not session_needs_reset(live, registry.device_id) or (
        live == registry.last_reset
    ):
        # Either the session already describes the live device, or a
        # successful reset already ran against this exact live read and the
        # session is bound to that device. Without the second arm, the
        # "doubt falls to not-alive" fail-safe would pay a full terminate-all
        # + Pa_Initialize + re-enumeration on EVERY press -- on the keep-warm
        # path, synchronously in the hotkey callback. The compared pair comes
        # from one read, so the guard checks it against the session it
        # guards; a moved or revived default is a different pair and resets,
        # and a failed reset clears the pair (open() writes None), so failed
        # resets retry.
        #
        # Reaching this line is itself the evidence the rebuild budget was
        # asking for -- the session is bound to what the system is on -- so
        # the streak and the verdict it produced go with it. While the
        # backoff is open it does not: a liveness flag flapping through a
        # storm reads as "settled" on every other press, and taking that as
        # the clearing evidence would hand the budget back to the very
        # device that spent it and let it cost a full teardown every other
        # press for the life of the process.
        if not _teardown_backoff_open(registry):
            registry.rebuild.settle()
            registry.needs_restart = False
            # The verdict describes the teardown as it is NOW (see
            # restart_verdict), so the refusal budget it was counted against
            # goes with it: leaving terminate_failures standing meant one
            # later refusal re-declared the verdict on sight, silently
            # retiring MAX_TERMINATE_FAILURES as a budget after the first
            # settle.
            registry.terminate_failures = 0
        return unchanged(registry)

    if _teardown_backoff_open(registry) and not _press_is_new_evidence(registry, live):
        # The press that still wants the teardown is the one the verdict is
        # about, and re-running it now stalls this press for a result this
        # session already has. Sitting below the live read and the dedupe
        # check is what makes the two refusals differ: a session bound to
        # what the system reports, and a user who moved the default to a
        # different device, are both picked up on that press instead of
        # being refused for the rest of the window without the device ever
        # having been read.
        return unchanged(registry, busy=True)
    return None


def _native_blocker(
    registry: PyAudioRegistry,
    warm: WarmCapture | None,
    blocked_workers: Callable[[], Sequence[threading.Thread | None]] | None,
) -> str | None:
    """Why PortAudio may not be driven to zero on this press, or None.

    Never drive PortAudio's init count to zero while another thread may
    be inside a native open() or stream.read() on a registered instance:
    that is the state cleanup() refuses to terminate in (issue #37). The
    ledger (registry.workers / registry.warm) is the
    authority -- it names every worker this session started, including an
    issue #60 orphan whose name its successor already cleared, and the
    capture installed in the arm whether or not this caller handed one
    over. blocked_workers is the belt to that braces: the threads the
    caller joined and can speak for, resolved through the callable at
    GATE time so a finished W1 cannot mask a live W2. The thread running
    this reset is never its own blocker.
    """
    blocked = terminate_blocker(registry.workers, registry.warm, exempt=warm)
    if blocked is None and worker_may_be_native(blocked_workers):
        return "named recording worker"
    return blocked


def _rebuild(
    registry: PyAudioRegistry,
    live: LiveDefaultInput,
    warm: WarmCapture | None,
    rearm_warm: Callable[[bool], None] | None,
    find_default_input_device: Callable[[pyaudio.PyAudio], int],
) -> SessionRefresh:
    """Tear the engine down and build the session again on the live device.

    The caller has proved nothing native is in flight and the keep-warm
    capture is stopped, so PortAudio's init count can be driven to zero here.
    Busy, not half-reset: nothing here is torn down on a bail-out, so the
    session stays dirty and the next press asks again.
    """
    registry.pyaudio = None
    registry.forget_index()
    rebuilt = registry.open(find_default_input_device)
    if not rebuilt:
        # The engine never came back. Leave the session unbound -- the next
        # press re-reads the live device and retries, with nothing left
        # holding PortAudio open. Nothing is re-armed over a session that
        # does not exist; the recovery ask brings keep-warm back when the
        # engine does.
        _hand_back_capture(registry, warm, rearm_warm, ask_recovery=True)
        # Nothing was rebuilt: the session is unbound, and an index into an
        # instance that no longer exists is not an index.
        return SessionRefresh(pyaudio=None, input_device_index=None, busy=False)
    if registry.last_reset is None:
        # open()'s own read gave no evidence (transient -- quite possible
        # right after the coreaudiod storm that moved the default). Record
        # the pair THIS read saw instead: the guard needs a baseline, and
        # whichever read it comes from, the pair it compares has to be the
        # one read that set it. Written BEFORE the hand-back below: the
        # capture the worker's recovery re-arms can run a reset from its own
        # thread immediately, and finding the baseline still unset there
        # costs a second full rebuild inside this same press -- and a second
        # unsynchronised write to the budget.
        registry.last_reset = live
    if warm is not None:
        # The press that moved the default records on the WORKER, not on a
        # capture armed here: a fresh WarmCapture has not opened its stream
        # yet (its run loop sets _ready and seeds the ring only after
        # _open_stream() returns), so routing this press through begin()
        # seeds the take from an empty ring -- the press comes back empty
        # and mislabelled "microphone busy" for a reset that succeeded.
        # Hand the stopped capture back (dead for presses either way) and
        # ask for recovery: the press's own worker brings keep-warm back
        # through WarmKeep.recover(), and the new capture's open joins that
        # worker before opening, so #16's no-overlapping-opens rule holds.
        _hand_back_capture(registry, warm, rearm_warm, ask_recovery=True)
    # An AirPod whose liveness flag oscillates makes every read a different
    # pair, and a default that alternates between two devices makes every
    # read a different DEVICE; the dedupe guard above is structurally blind
    # to both, and every press pays a full teardown + Pa_Initialize +
    # re-enumeration on the hotkey callback thread. Reaching here IS the
    # churn the budget exists to bound: counted after the rebuild because a
    # clean teardown resets the verdict, and this one earned it. The
    # same-device streak says WHICH device never settled (and its message
    # names it); the cross-device count bounds the default that alternates,
    # which restarts any per-device count on every press.
    registry.rebuild.note_rebuild(live.device_id)
    if registry.rebuild.streak >= MAX_RESET_REBUILDS:
        _declare_needs_restart(
            registry,
            f"Input device {live.device_id} never settled after "
            f"{registry.rebuild.streak} rebuilds; the microphone is not "
            "settling.",
        )
    elif registry.rebuild.count >= MAX_RESET_REBUILDS:
        _declare_needs_restart(
            registry,
            f"The input default kept moving: {registry.rebuild.count} "
            "rebuilds without the session ever agreeing with it; the "
            "microphone is not settling.",
        )
    return unchanged(registry)


# Re-exported for callers that name the entry point rather than the module.
__all__ = [
    "MAX_TERMINATE_FAILURES",
    "MICROPHONE_NEEDS_RESTART",
    "MICROPHONE_NOT_SETTLING",
    "SessionRefresh",
    "note_terminate_outcome",
    "reset_session",
    "restart_verdict",
]
