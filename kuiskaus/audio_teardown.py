"""How ``terminate_all()`` takes one registered instance down (issue #72).

Split from ``kuiskaus.audio_device``, which owns the instances themselves and
already owns "may I terminate at all" (the ledger gate). This is the other
half: in what ORDER one instance leaves the registry when the teardown runs.

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

A refusal keeps the old observable state: the instance that would not
terminate goes back on the list (an untracked live instance is unreachable,
the rule ``terminate_all()`` is built on) and its index binding -- still a
fact about a still-live instance -- is republished.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kuiskaus.audio_retry import terminate_quietly
from kuiskaus.audio_worker_keep import terminate_blocker

if TYPE_CHECKING:
    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry


def dispose_prepared(registry: PyAudioRegistry, pa: pyaudio.PyAudio) -> bool:
    """Untrack, unbind, then terminate one instance. True = it is gone.

    The untrack/unbind run BEFORE the native call so ``snapshot()`` can
    never pair this instance with a live index while ``Pa_Terminate()`` is
    in flight (module docstring). A refusal restores both: the instance
    stays reachable for the next attempt, and its binding stands because it
    is still live and the index is still a fact about it.
    """
    bound = registry._indices.resolve(pa)
    registry.forget(pa)
    registry._indices.unbind(pa)
    if terminate_quietly(pa):
        return True
    registry.register(pa)
    if bound is not None:
        registry._indices.publish(bound, pa)
    return False


def refuse_teardown(registry: PyAudioRegistry) -> bool:
    """True (and logged) while something this session started may still
    be inside a native PortAudio call.
    """
    blocker = terminate_blocker(registry.workers, registry.warm)
    if blocker is None:
        return False
    print(f"{blocker.capitalize()} may still be native; not terminating PyAudio.")
    return True
