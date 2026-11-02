"""The reset's answer to its caller (issue #72).

Split from ``kuiskaus.audio_device_reset``, which owns the rebuild. The
module-size seam is the reason, but the seam is also a real one: this is the
CONTRACT between the reset and whoever entered it -- the press gate reads
``locked``/``busy``, the recorder writes the pair back under its lock --
while the rebuild reads none of it. It lives here so the two shapes can be
read without the four hundred lines of teardown around them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import pyaudio

    from kuiskaus.audio_device import PyAudioRegistry

    # Annotation alias: the SessionRefresh field below is named ``pyaudio``,
    # which would otherwise shadow the module inside its own class body (the
    # same alias audio_retry.py uses for the same reason).
    _PyAudio = pyaudio.PyAudio


class SessionRefresh(NamedTuple):
    """What the caller writes back under its lock: the session to use and
    the device index resolved against it. ``busy`` means something may still
    be inside a native call, so PortAudio was NOT terminated and the session
    stays dirty for the next press to retry. ``locked`` marks the busy that
    is only "another thread's reset is running right now" -- admissible to
    the press gate, unlike the ledger deferral."""

    pyaudio: _PyAudio | None
    input_device_index: int | None
    busy: bool
    locked: bool = False


def unchanged(
    registry: PyAudioRegistry, *, busy: bool = False, locked: bool = False
) -> SessionRefresh:
    """A refresh result that leaves the session where it is.

    The pair is read from ``snapshot()`` as one value: an index is only a
    fact about the instance that enumerated it.
    """
    pyaudio_instance, device_index = registry.snapshot()
    return SessionRefresh(pyaudio_instance, device_index, busy, locked)
