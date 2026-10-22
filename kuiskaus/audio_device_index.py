"""A device index and the instance whose enumeration produced it (issue #72).

PortAudio snapshots the hardware once at ``Pa_Initialize`` and never
refreshes it, so a device index is not a fact about the machine -- it is a
fact about ONE ``PyAudio`` instance's frozen device list. A value resolved
on instance A names a device that instance B may not have; carrying it over
opens the stream somewhere that does not exist.

That is the shape of the wedge this class closes: a retry attempt on an
unbound session resolved the default input device, published it, and then
failed and disposed the very instance the index came from. The published
value survived, every later attempt trusted it, skipped re-resolution, and
opened on a dead index until the process restarted -- the mic unavailable
with no recovery path, inside the exact failure domain issue #72 is about.

So an index is stored WITH its instance, and read only while that instance is
the live one. A binding whose instance is gone reads back as "unresolved",
which is precisely the state that makes the next attempt enumerate again.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyaudio


class DeviceIndexBinding:
    """The session's device index, paired with its instance.

    Owned by ``PyAudioRegistry``, which owns the live instance it pairs
    against. The (index, owner) pair is stored as ONE tuple, so a reader
    takes both halves from the same value in a single load. Two fields could
    not make that true in any write order: a reader that sampled the owner
    and then loaded the index would see a NEW index attributed to an owner
    that never resolved it -- the wedge this class exists to close, arrived
    at through the read. Storing one tuple makes the claim structural rather
    than a comment about write ordering, and keeps ``resolve`` a plain
    identity test against the live instance.

    Threading: one small lock makes the read-modify-writes atomic. It exists
    for ``unbind`` alone -- "forget the index if it came from this instance"
    is a test and a clear that have to be one step, or a release racing a
    bind clears a binding that had already moved on to a live instance.
    ``resolve`` stays lock-free: a single tuple load is already atomic, and
    taking the lock there would put a wait on the path every retry attempt
    reads through.
    """

    def __init__(self) -> None:
        self._pair: tuple[int | None, pyaudio.PyAudio | None] = (None, None)
        self._lock = threading.Lock()

    def resolve(self, live: pyaudio.PyAudio | None) -> int | None:
        """The index for ``live``, or None when it does not describe it.

        None is the answer a caller wants when the pairing is stale: it
        means "re-resolve", which is the only correct response to an index
        that names a device list nobody here has.
        """
        index, owner = self._pair
        if live is None or owner is not live:
            return None
        return index

    def bind(self, device_index: int, resolved_on: pyaudio.PyAudio) -> None:
        """Publish an index a caller just resolved against ``resolved_on``.

        The caller (``PyAudioRegistry.bind_index``) checks first that no
        binding is already serving the live instance -- this class cannot:
        it holds the owner, but the live instance lives in the registry.

        One store of one tuple: see the class docstring for why the pair has
        to BE the value rather than two fields written in some order.
        """
        with self._lock:
            self._pair = (device_index, resolved_on)

    def publish(self, device_index: int | None, live: pyaudio.PyAudio | None) -> None:
        """Write a (instance, index) pair a caller holds together (issue #72).

        The recorder's session write-back comes through here: the pair it
        holds was read as a pair, so the pairing is recorded as it was read
        rather than inferred afterwards. One store of one tuple (class
        docstring); an index of None is no binding at all, so it carries no
        owner either.
        """
        with self._lock:
            self._pair = (
                device_index,
                live if device_index is not None else None,
            )  # fmt: skip

    def unbind(self, gone: pyaudio.PyAudio) -> None:
        """Forget the index if it came from ``gone``.

        Called when an instance is terminated: the device list it enumerated
        no longer exists, so an index into it must not survive as if it did.
        An instance that would not terminate is NOT passed here -- it is
        still live, and the binding is still a fact about it.
        """
        with self._lock:
            if self._pair[1] is gone:
                self._pair = (None, None)

    def clear(self) -> None:
        """No index, no owner: nothing is resolved."""
        with self._lock:
            self._pair = (None, None)
