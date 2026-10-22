"""Every recording worker this session built, and what it still blocks.

``PyAudioRegistry.terminate_all()`` drives PortAudio's init count to zero,
and doing that while a thread sits inside ``Pa_OpenStream`` is undefined
behaviour -- it is the state ``cleanup()`` exists to refuse (issue #37
task-c). The reset gate asked its CALLER to name those threads, and the
caller only ever had two names to give: the worker it joined and the one it
is starting. A worker wedged inside ``pa.open()`` (issue #60's orphan) stops
being nameable the moment its successor clears ``recording_thread``, so four
ordinary presses reached the terminate (issue #72 adversarial review,
CRITICAL): the wedge outlives the join budget, the successor completes, the
next press is told nothing is alive, and ``Pa_Terminate`` frees the state
the wedged worker is still entering.

Warm captures solved the same problem with a ledger (``WarmKeep``) precisely
because "the invariant is only checkable while somebody still holds the
thread". This is that ledger for workers: the session records each worker it
spawns, so liveness is checkable without anybody remembering a name.

Held strongly, never weakly: a thread PortAudio is still inside must not be
collectable while a terminate is asking about it, and ``threading.Thread``
supports no weak reference while it is running.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kuiskaus.audio_warm import WarmCapture
    from kuiskaus.audio_warm_keep import WarmKeep


class WorkerKeep:
    """The workers this session started, and whether one may be native.

    Owned by ``PyAudioRegistry``, which owns what they must not be native
    on. Threading: written by the press that spawns a worker, read by
    whichever thread is about to terminate PortAudio -- so every field move
    is under ``_lock``, and the lock is never held across anything that can
    block (``is_alive()`` is a flag read, never a join).
    """

    def __init__(self) -> None:
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()

    def note(self, worker: threading.Thread) -> None:
        """Record a worker the caller is about to start.

        Finished threads are pruned on the way in rather than left to
        accumulate: the ledger exists to answer one question -- is anybody
        still inside PortAudio -- and a completed worker does not participate
        in the answer, so keeping it would only grow a list that every
        terminate has to walk."""
        with self._lock:
            self._workers = [t for t in self._workers if t.is_alive()]
            self._workers.append(worker)

    def any_may_be_native(self) -> bool:
        """True when a recorded worker may still be inside a native call.

        The thread asking is never its own blocker: a worker runs the reset
        for its own press, having joined its predecessor first, and would
        otherwise refuse every reset it ever attempted."""
        me = threading.current_thread()
        with self._lock:
            pending = list(self._workers)
        return any(t is not me and t.is_alive() for t in pending)


def terminate_blocker(
    workers: WorkerKeep,
    warm: WarmKeep,
    exempt: WarmCapture | None = None,
) -> str | None:
    """Why PortAudio may NOT be terminated right now, or None if it may.

    The three things that can be inside a native call on a registered
    instance, asked in one place (issue #72 adversarial review): a worker
    this session started -- named or not, which is the point -- a keep-warm
    capture still installed in the arm, and a capture this session dropped
    while its read was wedged. The verdict is a label so the caller can say
    what it found, not just that it found something.

    ``exempt`` names the capture a caller is about to stop itself. Without it
    a keep-warm press could never reset: the capture it is handing over is
    still installed when the gate runs, and would block the very reset that
    stops it.
    """
    if workers.any_may_be_native():
        return "recording worker"
    if warm.installed_may_be_native(exempt=exempt):
        return "keep-warm capture"
    if not warm.clear():
        return "dropped keep-warm capture"
    return None
