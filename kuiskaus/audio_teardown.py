"""How ``terminate_all()`` takes the session's instances down (issue #72).

Split from ``kuiskaus.audio_device``, which owns the instances themselves and
already owns "may I terminate at all" (the ledger gate). This is the other
half: the whole ``terminate_all()`` pass, and in what ORDER one instance
leaves the registry when it runs (the registry's ``terminate_all()`` is the
delegation into it, the same seam ``refresh()`` uses for the device reset).

The order is the bug this module exists for. ``_dispose`` called
``terminate()`` FIRST and only then ``forget``/``unbind``, so while the native
``Pa_Terminate()`` was in flight the instance was still registered and its
index binding still resolved -- and ``snapshot()``, the one read the retry
loop takes to get its (instance, index) pair, returned exactly the
(corpse, live index) pairing ``DeviceIndexBinding`` exists to make
unrepresentable. A worker that snapshotted inside that window opened on an
instance PortAudio had already torn down. Forgetting and unbinding BEFORE the
terminate closes the window: no read can pair an instance that has been
terminated with an index that still resolves.

Untracking is the exclusive claim on the teardown. ``terminate_all()`` takes
no recorder lock and no ``_reset_lock``, so the cleanup path and a reset can
be in it at the same moment, both holding a snapshot of the same instance;
forgetting first made the pairing safe without ever making the terminate
unique, and on real pyaudio a second ``Pa_Terminate()`` on one instance drops
PortAudio's init count below the live reference count and tears down an
unrelated live instance (-10000). So ``forget()`` answers whether THIS call
removed the instance, and the one that did not own its teardown either.

A refusal keeps the old observable state: the instance that would not
terminate goes back on the list (an untracked live instance is unreachable,
the rule ``terminate_all()`` is built on) and its index binding -- still a
fact about a still-live instance -- is republished THROUGH THE GUARD, because
a press may have resolved and bound a fresher index while the native call was
in flight, and restoring a value read before it would make the stale one win.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kuiskaus.audio_device_reset import note_terminate_outcome
from kuiskaus.audio_retry import terminate_quietly
from kuiskaus.audio_worker_keep import terminate_blocker

if TYPE_CHECKING:
    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry


def dispose_claimed(registry: PyAudioRegistry, pa: pyaudio.PyAudio) -> bool:
    """Unbind, then terminate one instance the caller has ALREADY untracked.
    True = it is gone.

    The unbind runs BEFORE the native call so ``snapshot()`` can never pair
    this instance with a live index while ``Pa_Terminate()`` is in flight
    (module docstring). A refusal restores both: the instance goes back on
    the list, reachable for the next attempt, and its binding stands because
    it is still live and the index is still a fact about it -- unless
    something bound a fresher one in the meantime, which the guarded write
    below defers to rather than overwriting.

    Two callers, two ownership stories. ``dispose_prepared`` is
    ``terminate_all()``'s: it claims the instance out of the shared list
    first, so a second concurrent pass cannot take the same instance down
    twice. ``PyAudioRegistry._dispose`` is the caller-that-built-it path (the
    retry loop's own cleanup, the instance a reset displaced) and reaches this
    function directly, having untracked the instance itself -- nothing else
    can be reaching for an instance nobody else has, so there is no race to
    claim away, and gating that path on a claim would let a live untracked
    instance survive unterminated with its binding still naming a device it
    never enumerated.
    """
    bound = registry._indices.resolve(pa)
    registry._indices.unbind(pa)
    if terminate_quietly(pa):
        return True
    registry.register(pa)
    if bound is not None:
        # ``bind`` and not ``publish``: the refusal restores a value read
        # before the native call, so it may no longer be the newest fact
        # about this instance. A binding that arrived meanwhile -- against
        # the instance the session is on -- is the newer fact and stands.
        registry._indices.bind(bound, pa, registry.pyaudio)
    return False


def dispose_prepared(registry: PyAudioRegistry, pa: pyaudio.PyAudio) -> bool:
    """Claim one instance, then take it down (``dispose_claimed``).

    True = it is gone -- which includes "someone else claimed it and is
    terminating it": a caller that lost the claim has nothing left to do
    here, and reaching ``terminate()`` anyway is the double ``Pa_Terminate()``
    this module exists to prevent.

    This is the path ``terminate_all()`` uses, because it is the one that
    reads a list two threads can be reading at the same moment -- and the
    claim is what makes a second pass over the same snapshot a no-op rather
    than a second ``Pa_Terminate()``.
    """
    if not registry.forget(pa):
        return True
    return dispose_claimed(registry, pa)


def terminate_all_pass(registry: PyAudioRegistry) -> bool:
    """Terminate every instance the session ever built. True = all gone.

    The pass ``PyAudioRegistry.terminate_all()`` delegates here (the
    module-size seam, the same split ``refresh()`` uses for the device reset).

    PortAudio's init count is reference-counted, so a single missed instance
    -- an unregistered one, or one whose terminate() raised -- keeps the engine
    up and the next construction reuses the frozen device list.
    ``dispose_prepared`` claims each instance out of the shared list (module
    docstring), so a second concurrent pass over the same snapshot terminates
    nothing a second time.

    Two different False answers live here, and only one is counted. A refusal
    from the ledger gate -- a worker this session started, or a keep-warm
    capture it installed or dropped, may still be inside a native call -- is a
    deferral that releases itself, so callers treat it as busy and nothing is
    claimed. A refusal from an instance that would not terminate is retried
    and counted, and past ``MAX_TERMINATE_FAILURES`` in a row the session says
    so through ``needs_restart`` rather than leaving the app reporting Ready
    over a dead microphone.
    """
    if registry._refuse_teardown():
        return False
    incomplete = False
    disposed = 0
    with registry._instances_lock:
        pending = list(registry._instances)
    # The ledger gains names on other threads (a press notes its worker under
    # the recorder's lock, not this one), so the gate is asked again
    # immediately before anything is disposed: the snapshot above is a list of
    # instances, not a promise about the threads on them.
    if registry._refuse_teardown():
        return False
    for pa in pending:
        if dispose_prepared(registry, pa):
            disposed += 1
        else:
            incomplete = True
    if incomplete and disposed:
        # Say so when the pass was only half a teardown: the session can be
        # left pointing at an instance this call DID terminate (a press then
        # opens on it and gets "microphone unavailable" on healthy hardware
        # until the next press re-binds), and without this line that press has
        # no trace of why.
        print(
            f"Terminated {disposed} of {len(pending)} PyAudio instances; "
            "the session may still name one of them."
        )
    live = registry.pyaudio
    if live is not None and not registry.tracks(live):
        # The session names an instance this pass terminated. Left bound, the
        # retry loop opens on a corpse, so the name goes with the instance and
        # the next press rebuilds. Unbind rather than clear: a press that
        # rebuilt in the meantime owns the binding now.
        registry._indices.unbind(live)
        if registry.pyaudio is live:
            registry.pyaudio = None
    if not pending:
        # Nothing was tracked, so nothing was torn down: an empty pass is not
        # a teardown that came clean, and booking it as one retires a restart
        # verdict some earlier, real teardown earned -- a press that never
        # touched PortAudio would clear the claim against a dead microphone.
        return True
    note_terminate_outcome(registry, clean=not incomplete)
    return not incomplete


def refuse_teardown(registry: PyAudioRegistry) -> bool:
    """True (and logged) while something this session started may still
    be inside a native PortAudio call.
    """
    blocker = terminate_blocker(registry.workers, registry.warm)
    if blocker is None:
        return False
    print(f"{blocker.capitalize()} may still be native; not terminating PyAudio.")
    return True
