"""Keep-warm captures this session dropped, and what they still block.

A keep-warm capture whose thread is parked in a native ``stream.read()``
cannot be stopped, and its open stream keeps a PortAudio instance's init
count above zero. A recorder that keeps such a capture installed records the
next press into a sink that goes silent mid-take (its stop event is already
set, so it can never re-open); a recorder that simply forgets it loses the
only handle on a thread that is still reading. Either mistake ends the same
way -- ``PyAudio.terminate()`` running underneath a live reader -- so the
capture is dropped from the recorder's state and named here instead, where
every later terminate can refuse to run until its thread has exited.

This is the issue #72 review's CRITICAL: the invariant "never terminate
PortAudio while a warm thread may be inside a native read" is only checkable
while somebody still holds the thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from kuiskaus.audio_warm import WarmCapture


class WarmKeep:
    """The keep-warm request, and the captures dropped while reading it.

    Owned by the session (``PyAudioRegistry``), because the session is what
    must not be terminated while any of them may still be reading.
    """

    def __init__(self) -> None:
        # The menu's keep-warm request (issue #66), mirrored so a capture
        # dropped on a stuck read can be brought back.
        self.enabled = False
        self._blockers: list[WarmCapture] = []
        self._recover_pending = False

    def drop(self, warm: WarmCapture | None) -> bool:
        """Stop a capture the caller is letting go of. True = nothing of it
        is left that could still be reading.

        ``stop()`` returning False means the capture's thread is alive by
        definition -- parked in a native ``stream.read()`` -- with its stream
        still open on a registered instance. The caller has just cleared its
        own reference, so recording it here is what keeps that thread
        nameable: what this replaces was dropping it, which let the next
        press or cleanup() drive PortAudio's init count to zero underneath it.
        """
        if warm is None or warm.stop():
            return True
        if warm not in self._blockers:
            self._blockers.append(warm)
        self._recover_pending = True
        return False

    def clear(self) -> bool:
        """True when nothing dropped here may still be inside a native
        PortAudio call -- so terminating PortAudio is safe.

        Probing is how a blocker is released: ``stop(0.0)`` re-sets the
        (already set) stop event and joins with no timeout, so it reports
        liveness without ever parking the caller on a thread that may be
        parked for the life of the process.
        """
        live: list[WarmCapture] = []
        for warm in self._blockers:
            if not warm.stop(0.0):
                live.append(warm)
        self._blockers = live
        return not live

    def want(self, enabled: bool) -> None:
        """Record the menu's keep-warm request (issue #66).

        Checking it also withdraws a pending recovery: a capture the user
        switched off and then back on must not wait on a stale flag.
        """
        self.enabled = enabled
        if enabled:
            self._recover_pending = False

    def need_recovery(self) -> None:
        """Ask for keep-warm to be brought back later (issue #72 review).

        ``drop()`` sets the recovery flag only when ``stop()`` FAILS, so a
        reset that stopped its capture cleanly and then bailed (the
        ``terminate_all()`` refusal) left keep-warm off for the life of the
        process with the menu still showing the checkmark: nothing had set
        the flag, so ``recover()`` never fired. This is the explicit ask for
        exactly that case, and the worker path's ``recover()`` picks it up
        the same way it picks up a stuck read.
        """
        self._recover_pending = True

    def recover(self, rearm_warm: Callable[[bool], None] | None) -> None:
        """Bring keep-warm back after a stuck-read drop (issue #72 review).

        ``rearm_warm(True)`` is otherwise reachable only from a warm press
        that stopped its capture, so one stuck read left pre-roll off for the
        life of the process while the menu still showed the checkmark. The
        caller invokes this from the worker path, where the press has already
        committed to a worker: the new capture's own open joins that worker
        before opening, so the two native opens never overlap (issue #16).
        """
        if rearm_warm is None or not self.enabled:
            return
        if self._recover_pending and self.clear():
            self._recover_pending = False
            rearm_warm(True)
