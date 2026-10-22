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

This is the issue #72: the invariant "never terminate
PortAudio while a warm thread may be inside a native read" is only checkable
while somebody still holds the thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from kuiskaus.audio_warm import WarmCapture


def _stop(warm: WarmCapture) -> bool:
    """``warm.stop()`` with the foreign-call failure handled.

    ``WarmCapture.stop()`` joins a thread this session does not control, and
    a raising join is exactly as much evidence of "still reading" as a False
    return: the caller must assume the thread is live and keep PortAudio up.
    Reporting success here would be the unsafe direction.
    """
    try:
        return warm.stop()
    except Exception as e:  # noqa: BLE001 - foreign capture boundary, logged
        print(f"Keep-warm capture stop raised ({str(e)[:120]}); treating as stuck.")
        return False


def _probe_stop(warm: WarmCapture) -> bool:
    """``warm.stop(0.0)`` with the same rule as ``_stop``: a raise means it
    may still be reading, so it stays on the ledger.
    """
    try:
        return warm.stop(0.0)
    except Exception as e:  # noqa: BLE001 - foreign capture boundary, logged
        print(f"Keep-warm capture probe raised ({str(e)[:120]}); treating as stuck.")
        return False


class WarmKeep:
    """The keep-warm request, and the captures dropped while reading it.

    Owned by the session (``PyAudioRegistry``), because the session is what
    must not be terminated while any of them may still be reading.

    Threading (issue #72): this list IS the check the invariant rests on,
    and it has more than one writer. A press's
    ``drop()`` runs on the hotkey dispatcher's worker thread while
    ``set_keep_warm``/``cleanup()`` run on the rumps main thread, and
    ``CallbackDispatcher.stop(timeout=5.0)`` explicitly tolerates an in-flight
    press callback outliving its join -- so a stuck press and a quit really
    can overlap. Every field is therefore read and written under ``_lock``.
    The lock is never held across ``warm.stop()``: that join can take seconds
    and a probe from the other thread must be answered immediately rather
    than queued behind it, which is why ``clear()`` snapshots, probes
    lock-free, then commits by removal (see each method).
    """

    def __init__(self) -> None:
        # The menu's keep-warm request (issue #66), mirrored so a capture
        # dropped on a stuck read can be brought back.
        self.enabled = False
        self._blockers: list[WarmCapture] = []
        # The capture the recorder has INSTALLED right now. ``_blockers``
        # only ever held captures somebody had already handed over to drop(),
        # so a capture armed and never dropped was invisible to
        # terminate_all(): the
        # worker-side reset passed no capture at all, resting on "with
        # keep-warm on there is no worker" -- a race, not an invariant, since
        # WarmArm.set_enabled() arms from the menu thread. The name lives
        # here as well as in WarmArm so the session can answer the question
        # without a caller handing it the answer.
        self._installed: WarmCapture | None = None
        self._recover_pending = False
        # Guards the fields above. Never held across warm.stop().
        self._lock = threading.Lock()

    def note_installed(self, warm: WarmCapture | None) -> None:
        """Mirror the capture the recorder just installed (or cleared).

        Called from ``WarmArm`` under the recorder lock; the name is taken
        BEFORE the capture's thread starts, so a terminate racing the arm
        sees the capture it must not terminate underneath."""
        with self._lock:
            self._installed = warm

    def installed_may_be_native(self, exempt: WarmCapture | None = None) -> bool:
        """True while the installed keep-warm capture has a live thread -- or
        has not been started yet.

        Two names are excused. The capture's own thread is never its own
        blocker: WarmCapture opens and reads on that thread, and it is also
        the thread that runs the keep-warm session reset (issue #72) --
        counting it would refuse every reset the capture ever asked for. And
        ``exempt`` is the capture a caller is on its way to stop, which the
        warm press hands over before this is asked.

        No thread yet means ARMING, not idle (issue #72): ``WarmArm`` names
        the capture here precisely so a
        terminate racing the arm sees it, and it does so BEFORE
        ``capture.start()`` assigns the thread. Requiring a non-None thread
        made this predicate blind to the one window ``note_installed`` exists
        to cover, so ``terminate_all()`` -- which takes no recorder lock --
        could read the ledger in that gap and drive PortAudio's init count to
        zero underneath a capture about to open on it. The conservative
        direction is the safe one: an installed capture nobody has dropped is
        a blocker until its thread is provably not running.
        """
        with self._lock:
            warm = self._installed
        if warm is None or warm is exempt:
            return False
        thread = warm.thread
        if thread is None:
            return True
        return thread is not threading.current_thread() and thread.is_alive()

    def drop(self, warm: WarmCapture | None) -> bool:
        """Stop a capture the caller is letting go of. True = nothing of it
        is left that could still be reading.

        ``stop()`` returning False means the capture's thread is alive by
        definition -- parked in a native ``stream.read()`` -- with its stream
        still open on a registered instance. The caller has just cleared its
        own reference, so recording it here is what keeps that thread
        nameable: what this replaces was dropping it, which let the next
        press or cleanup() drive PortAudio's init count to zero underneath it.

        The name goes on the list BEFORE ``stop()`` and comes off only when
        ``stop()`` reports success (issue #72). Registering
        after the join instead left that whole window -- precisely the
        interval in which the thread is proven wedged in ``stream.read()``
        -- with the capture invisible, so a ``clear()`` racing in from the
        menu thread reported a safe-to-terminate session while the reader was
        still parked on it.

        Dropping the same capture twice is idempotent about the JOIN (issue
        #72): a capture already on the blockers list has been
        joined once and is proven parked, so the second caller gets the
        non-blocking probe. Without that, ``refresh()`` dropping a wedged
        capture and then handing the same object to ``rearm_warm(False)`` --
        which drops it again -- cost TWO 2.0s joins of a thread that cannot
        answer, on the hotkey dispatcher's own thread (~4s for one refused
        press, measured): the press-path stall class issue #60 exists to keep
        off this thread.
        """
        if warm is None:
            return True
        with self._lock:
            hand_back = warm in self._blockers
            if not hand_back:
                self._blockers.append(warm)
            # It is no longer the installed capture either way: stopped, it
            # is gone; wedged, _blockers now carries the name. Leaving it in
            # both places would have installed_may_be_native() refuse every
            # later terminate on a capture the recorder has forgotten.
            if self._installed is warm:
                self._installed = None
        # An unlisted capture has never been joined here, so ask it to stop
        # for real; a listed one is being handed back -- probe, never re-join.
        stopped = _probe_stop(warm) if hand_back else _stop(warm)
        if stopped:
            with self._lock:
                if warm in self._blockers:
                    self._blockers.remove(warm)
            return True
        with self._lock:
            self._recover_pending = True
        return False

    def clear(self) -> bool:
        """True when nothing dropped here may still be inside a native
        PortAudio call -- so terminating PortAudio is safe.

        Probing is how a blocker is released: ``stop(0.0)`` re-sets the
        (already set) stop event and joins with no timeout, so it reports
        liveness without ever parking the caller on a thread that may be
        parked for the life of the process.

        The probes run with no lock held (they are foreign code, and a
        wedged capture must never be able to hold up the other thread's
        probe), so the commit removes the provably-dead entries from the
        CURRENT list rather than reassigning a snapshot -- a reassignment
        would permanently erase a capture that another thread's ``drop()``
        registered while these probes were running (issue #72).
        """
        with self._lock:
            pending = list(self._blockers)
        dead = [warm for warm in pending if _probe_stop(warm)]
        if not dead:
            with self._lock:
                return not self._blockers
        with self._lock:
            self._blockers = [warm for warm in self._blockers if warm not in dead]
            return not self._blockers

    def want(self, enabled: bool) -> None:
        """Record the menu's keep-warm request (issue #66).

        Checking it also withdraws a pending recovery: a capture the user
        switched off and then back on must not wait on a stale flag. Written
        from the menu thread while a worker thread may be mid-``drop()``, so
        both fields move under the lock together.
        """
        with self._lock:
            self.enabled = enabled
            if enabled:
                self._recover_pending = False

    def wanted(self) -> bool:
        """The user's keep-warm request AS IT STANDS NOW (issue #72).

        Read under the lock, not as a bare attribute read, because the
        question is asked by a thread that has been busy for as long as a
        full device reset -- terminate-all + ``Pa_Initialize`` + re-enumerate,
        hundreds of milliseconds -- during which the menu thread may have
        unchecked the item. A reset that then armed unconditionally resurrected
        the mic behind an unchecked menu.
        """
        with self._lock:
            return self.enabled

    def note_armed(self) -> None:
        """Record that a capture is now installed for the current request.

        A pending recovery is spent by it: leaving the flag set would have
        the next worker press re-arm a SECOND capture over a working one (the
        pre-fix callers reached this through ``want(True)``, which cleared it).
        """
        with self._lock:
            self._recover_pending = False

    def need_recovery(self) -> None:
        """Ask for keep-warm to be brought back later (issue #72).

        ``drop()`` sets the recovery flag only when ``stop()`` FAILS, so a
        reset that stopped its capture cleanly and then bailed (the
        ``terminate_all()`` refusal) left keep-warm off for the life of the
        process with the menu still showing the checkmark: nothing had set
        the flag, so ``recover()`` never fired. This is the explicit ask for
        exactly that case, and the worker path's ``recover()`` picks it up
        the same way it picks up a stuck read.
        """
        with self._lock:
            self._recover_pending = True

    def recover(self, rearm_warm: Callable[[bool], None]) -> None:
        """Bring keep-warm back after a stuck-read drop (issue #72).

        ``rearm_warm(True)`` is otherwise reachable only from a warm press
        that stopped its capture, so one stuck read left pre-roll off for the
        life of the process while the menu still showed the checkmark. The
        caller invokes this from the worker path, where the press has already
        committed to a worker: the new capture's own open joins that worker
        before opening, so the two native opens never overlap (issue #16).
        """
        with self._lock:
            requested = self.enabled and self._recover_pending
        if requested and self.clear():
            with self._lock:
                self._recover_pending = False
            rearm_warm(True)
