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

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from kuiskaus.audio_coreaudio import (
    LiveDefaultInput,
    read_live_default_input,
    session_needs_reset,
)
from kuiskaus.audio_rebuild_budget import MAX_RESET_REBUILDS
from kuiskaus.audio_retry import microphone_unavailable
from kuiskaus.audio_warm_arm import worker_may_be_native
from kuiskaus.audio_worker_keep import terminate_blocker

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

MICROPHONE_BUSY = "microphone busy — recording did not start"

# How many consecutive teardowns an instance may refuse before the session
# stops implying the microphone might come back (issue #72).
# Retrying forever is right -- forgetting a live instance would leave
# PortAudio's init count above zero and make every later rebuild useless --
# but retrying forever *silently* is what reproduced the issue's own symptom
# inside the fix: busy on every press, device_id pinned to a dead device,
# last_error None, the menubar showing Ready. Three is "the storm is not
# transient", and it still leaves room for one.
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
        # now told follows from ``terminate_failures`` being non-zero (see
        # ``restart_verdict``), so the stronger claim -- an instance really
        # did refuse to release -- replaces the weaker one on its own.
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
    teardown, so it belongs to the session that has a refusal on the record
    (``terminate_failures`` non-zero, which a clean teardown clears along
    with the verdict -- see ``note_terminate_outcome``). The streak verdict
    has no such refusal: every teardown released, and what will not settle is
    the device. Choosing the wording here rather than storing it is what
    keeps the sentence true after the state that earned it moves -- the same
    reason the streak itself is not frozen into a message.
    """
    if registry.terminate_failures:
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
    """
    return (
        not registry.terminate_failures
        and registry.rebuild.device is not None
        and registry.rebuild.is_new_device(live.device_id)
    )


class SessionRefresh(NamedTuple):
    """What the caller writes back under its lock: the session to use and
    the device index resolved against it. ``busy`` means something may still
    be inside a native call, so PortAudio was NOT terminated and the session
    stays dirty for the next press to retry."""

    pyaudio: pyaudio.PyAudio | None
    input_device_index: int | None
    busy: bool


def _unchanged(
    registry: PyAudioRegistry, *, busy: bool = False, unbound: bool = False
) -> SessionRefresh:
    """A refresh result that leaves the session where it is.

    The pair is read from ``snapshot()`` as one value: an index is only a
    fact about the instance that enumerated it.
    """
    if unbound:
        return SessionRefresh(pyaudio=None, input_device_index=None, busy=busy)
    pyaudio_instance, device_index = registry.snapshot()
    return SessionRefresh(
        pyaudio=pyaudio_instance, input_device_index=device_index, busy=busy
    )


def _hand_back_capture(
    registry: PyAudioRegistry,
    warm: WarmCapture | None,
    rearm_warm: Callable[[bool], None] | None,
    *,
    rearmable: bool,
    ask_recovery: bool,
) -> None:
    """Tell the recorder what became of the capture this reset stopped.

    A capture stop() reported stopped is dead for presses either way -- its
    stop event is set, so it can never re-open -- so it is handed back
    (``rearmable=False``) or re-armed (``rearmable=True``) rather than being
    left installed as a sink that silently swallows the next take.

    ``ask_recovery`` is the separate ask for the bail-out paths: stop()
    SUCCEEDED there, so ``drop()`` set no recovery flag and keep-warm would
    stay off -- menu checkmark and all -- for the life of the process.
    """
    if ask_recovery:
        registry.warm.need_recovery()
    if warm is None or rearm_warm is None:
        return
    rearm_warm(rearmable)


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
    """
    live = read_live_default_input()
    if live is None:
        # CoreAudio saying "no evidence": never treated as a change, and
        # never as an absence either. Nothing was proved about the session,
        # so nothing here may clear a verdict an earlier press earned.
        return _unchanged(registry)
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
        return _unchanged(registry)

    if _teardown_backoff_open(registry) and not _press_is_new_evidence(registry, live):
        # The press that still wants the teardown is the one the verdict is
        # about, and re-running it now stalls this press for a result this
        # session already has. Sitting below the live read and the dedupe
        # check is what makes the two refusals differ: a session bound to
        # what the system reports, and a user who moved the default to a
        # different device, are both picked up on that press instead of
        # being refused for the rest of the window without the device ever
        # having been read.
        return _unchanged(registry, busy=True)

    # Never drive PortAudio's init count to zero while another thread may
    # be inside a native open() or stream.read() on a registered instance:
    # that is the state cleanup() refuses to terminate in (issue #37
    # task-c). The ledger (registry.workers / registry.warm) is the
    # authority -- it names every worker this session started, including an
    # issue #60 orphan whose name its successor already cleared, and the
    # capture installed in the arm whether or not this caller handed one
    # over. blocked_workers is the belt to that braces: the threads the
    # caller joined and can speak for, resolved through the callable at
    # GATE time so a finished W1 cannot mask a live W2. The thread running
    # this reset is never its own blocker. Busy, not half-reset -- nothing
    # here is torn down, so the session stays dirty and the next press
    # asks again.
    blocked = terminate_blocker(registry.workers, registry.warm, exempt=warm)
    if blocked is None and worker_may_be_native(blocked_workers):
        blocked = "named recording worker"
    if blocked is not None:
        print(f"Device reset deferred: {blocked} may still be native.")
        return _unchanged(registry, busy=True)

    if not registry.warm.drop(warm):
        # The stuck read owns a capture whose stop event is already set:
        # when the read returns the thread exits and THIS capture can never
        # re-open. Left installed, the next press that needs no reset would
        # be admitted and record silently into a dead sink, so
        # rearm_warm(False) tells the recorder to drop it -- later presses
        # take the worker path.
        #
        # Dropped for presses, NOT gone for PortAudio: WarmKeep kept the
        # only remaining handle on that thread, and that handle is what
        # gates every later terminate_all() until the thread has actually
        # exited. The hand-back asks for that SAME object to be dropped
        # again; WarmKeep.drop() answers with a probe, not a second join.
        print("Keep-warm capture would not stop; dropping it.")
        _hand_back_capture(
            registry, warm, rearm_warm, rearmable=False, ask_recovery=False
        )
        return _unchanged(registry, busy=True)

    if not registry.terminate_all():
        # Busy: the session stays dirty and the next press retries. The
        # capture was stopped cleanly above and is dead for presses
        # either way, so hand it back and ask for recovery (see
        # _hand_back_capture).
        _hand_back_capture(
            registry, warm, rearm_warm, rearmable=False, ask_recovery=True
        )
        return _unchanged(registry, busy=True)
    registry.pyaudio = None
    registry.forget_index()
    rebuilt = registry.open(find_default_input_device)
    if not rebuilt:
        # The engine never came back. Leave the session unbound -- the next
        # press re-reads the live device and retries, with nothing left
        # holding PortAudio open. Nothing is re-armed over a session that
        # does not exist; the recovery ask brings keep-warm back when the
        # engine does.
        _hand_back_capture(
            registry, warm, rearm_warm, rearmable=False, ask_recovery=True
        )
        return _unchanged(registry, unbound=True)
    if registry.last_reset is None:
        # open()'s own read gave no evidence (transient -- quite possible
        # right after the coreaudiod storm that moved the default). Record
        # the pair THIS read saw instead: the guard needs a baseline, and
        # whichever read it comes from, the pair it compares has to be the
        # one read that set it. Written BEFORE the re-arm below: the re-armed
        # capture's own thread can run a reset immediately, and finding the
        # baseline still unset there costs a second full rebuild inside this
        # same press -- and a second unsynchronised write to the budget.
        registry.last_reset = live
    if warm is not None and rearm_warm is not None:
        # Only now does the session exist to arm against.
        rearm_warm(True)
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
    return _unchanged(registry)


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
    False = fail this press, and ``press_refusal_reason`` is what the user is
    told about it.
    """
    if recording or warm is None:
        return True
    return not reset_session(
        registry, find_default_input_device, warm, rearm_warm, lambda: (worker,)
    ).busy


def press_refusal_reason(registry: PyAudioRegistry) -> str:
    """Why an admitted press was refused (issue #72).

    The two refusals look alike to the press and are not alike. A ledger
    deferral -- a worker or a warm capture still inside a native call --
    releases itself, and "busy" is the honest word. A session whose teardown
    has refused ``MAX_TERMINATE_FAILURES`` in a row never does, and telling
    that user "busy" is exactly how the #72 symptom survived inside the PR
    written to remove it: busy on every press, last_error None, menubar
    Ready.
    """
    if registry.needs_restart:
        return restart_verdict(registry)
    return MICROPHONE_BUSY


def unavailable_reason(
    registry: PyAudioRegistry, last_error: BaseException | None
) -> str:
    """What an exhausted retry loop leaves in ``recorder.last_error``.

    The worker path has no press-side refusal to hang the restart verdict on
    -- its press is admitted and the retry loop is what fails -- so the same
    poisoned session surfaces here instead. While the teardown still has
    budget left this keeps ``microphone_unavailable``'s wording: a storm that
    clears must not be announced as an app restart. Once it does not, the
    underlying error still rides along -- "needs a restart" alone tells the
    user nothing about what the microphone actually did, and the errno is
    what makes a -9986 storm recognisable in a bug report.
    """
    if registry.needs_restart:
        verdict = restart_verdict(registry)
        if last_error is None:
            return verdict
        return f"{verdict} ({last_error})"
    return microphone_unavailable(last_error)


# Re-exported for callers that name the entry point rather than the module.
__all__ = [
    "MAX_TERMINATE_FAILURES",
    "MICROPHONE_BUSY",
    "MICROPHONE_NEEDS_RESTART",
    "MICROPHONE_NOT_SETTLING",
    "SessionRefresh",
    "note_terminate_outcome",
    "press_is_admissible",
    "press_refusal_reason",
    "reset_session",
    "restart_verdict",
    "unavailable_reason",
]
