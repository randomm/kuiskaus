"""What a refused press is told, and who is allowed to run the reset (issue #72).

Split from ``kuiskaus.audio_device_reset``, which owns the rebuild. That
module answers "what does it take to rebuild the session"; this one answers
the two questions a press asks on its way in and out: may it run the reset at
all, and when it may not (or its worker's retry loop fails), what does the
user read in the menubar's red line.

Kept separate because the answers are read by different callers under
different conditions: the gate runs before the recorder takes its lock, the
refusal text is read by the hotkey callback, and the retry-loop wording is
read by a worker whose press was already admitted. All three describe the
same poisoned session from three sides, which is why they live together
rather than beside the teardown they describe.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from kuiskaus.audio_device_reset import reset_session, restart_verdict
from kuiskaus.audio_retry import MICROPHONE_BUSY, microphone_unavailable

if TYPE_CHECKING:
    import threading

    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry
    from kuiskaus.audio_warm import WarmCapture

__all__ = [
    "press_is_admissible",
    "press_refusal_reason",
    "unavailable_reason",
]


def press_is_admissible(
    registry: PyAudioRegistry,
    recording: bool,
    warm: WarmCapture | None,
    worker: threading.Thread | None,
    find_default_input_device: Callable[[pyaudio.PyAudio], int],
    rearm_warm: Callable[[bool], None],
) -> bool:
    """The keep-warm press gate: run the live-default check, or say why this
    press must not run it.

    The gate lives next to the reset it protects: a press the recorder is
    about to refuse (``recording``) must NOT reach the reset -- it would stop
    the warm capture feeding that live recording and rebuild the session
    underneath it -- and a press with no warm capture has no press-side reset
    to run at all (its worker resets at the top of its own loop). Both are
    admitted (True). An admitted keep-warm press runs the reset itself;
    False = fail this press, and ``press_refusal_reason`` is what the user is
    told about it. A ``locked`` busy is the one busy that admits: it means
    the very reset this press asks for is already running on another thread
    (issue #72 -- press, worker and warm capture all enter), and a press must
    not fail because its own work is being done.
    """
    if recording or warm is None:
        return True
    result = reset_session(
        registry, find_default_input_device, warm, rearm_warm, lambda: (worker,)
    )
    return result.locked or not result.busy


def press_refusal_reason(registry: PyAudioRegistry) -> str:
    """Why an admitted press was refused.

    The two refusals look alike to the press and are not alike. A ledger
    deferral -- a worker or a warm capture still inside a native call --
    releases itself, and "busy" is the honest word. A session whose teardown
    has refused ``MAX_TERMINATE_FAILURES`` in a row never does, and telling
    that user "busy" leaves the app showing Ready over a microphone that will
    never come back.
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
        # Bounded like every other piece of foreign text in this area
        # (``str(arm_error)[:120]`` in the reset's hand-back): this string
        # lands in the menubar status item, and a native exception can carry
        # an arbitrarily long PortAudio/CoreAudio description.
        return f"{verdict} ({str(last_error)[:120]})"
    return microphone_unavailable(last_error)
