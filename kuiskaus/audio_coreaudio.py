"""What the system's default input device is RIGHT NOW (issue #72).

PortAudio snapshots the hardware once at ``Pa_Initialize`` and never
refreshes it, so a cached ``PyAudio`` keeps listing a device the system
no longer uses -- the stale "Built-in Microphone" behind issue #72. The
only honest answer comes from CoreAudio itself, so this module reads the
live default input device (and its liveness) with one-shot
``AudioObjectGetPropertyData`` calls: no property listeners and no
runloop, because on this machine ``AudioObjectAddPropertyListener``
blocks forever, and a listener-based design would reintroduce the
issue #60 hang with an unbounded thread.

``kuiskaus.audio_device`` (the registry/session reset) asks this module
the "which device am I on" half of its question and owns the "what has
to be rebuilt when it moves" half.

The CoreAudio constants are four-char codes and their names are the only
documentation of what a bare integer means, so they are named here.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

kAudioObjectSystemObject = 1
# The codes below are the PyObjC bridge's own values -- verified against
# kAudioHardwarePropertyDefaultInputDevice / kAudioDevicePropertyDeviceIsAlive
# on a real CoreAudio install. The four-char decodings are 'dIn ' and 'livn'
# (they are NOT 'defi'/'aliv'); the tests mirror the same integers, so a
# comment-following "fix" that retypes one of these numbers breaks live
# detection while the suite stays green.
kAudioHardwarePropertyDefaultInputDevice = 1682533920  # 'dIn '
kAudioDevicePropertyDeviceIsAlive = 1818850926  # 'livn'
kAudioObjectPropertyScopeGlobal = 1735159650  # 'glob'
kAudioObjectPropertyElementMain = 0
kAudioObjectUnknown = 0

# AudioObjectGetPropertyData's last argument is an in/out buffer whose
# address PyObjC must pin; passing None (rather than a preallocated
# bytearray) makes the bridge allocate it and hand the bytes back.
_OUT_BUFFER = None
_SIZE_BYTES = 4
_NO_QUALIFIER = b""


class LiveDefaultInput(NamedTuple):
    """The system default input device as CoreAudio sees it right now."""

    device_id: int
    is_alive: bool


def _read_uint32(object_id: int, selector: int, scope: int, element: int) -> int | None:
    """One UInt32 property read, or None if CoreAudio refused it.

    CoreAudio reports failure as a nonzero OSStatus rather than raising,
    so a None here means "no evidence" -- never "absent". But the PyObjC
    bridge around those calls CAN raise (marshalling under a coreaudiod
    storm, a first-call import failure), and this read runs on the
    AudioRecorder.__init__ and press paths, where an escaping exception
    is an app that never launches. Both failure shapes collapse to None."""
    try:
        import CoreAudio

        address = CoreAudio.AudioObjectPropertyAddress(selector, scope, element)
        status, _size, raw = CoreAudio.AudioObjectGetPropertyData(
            object_id, address, 0, _NO_QUALIFIER, _SIZE_BYTES, _OUT_BUFFER
        )  # fmt: skip
    except Exception as e:  # noqa: BLE001 - PyObjC/CoreAudio boundary, logged
        # Bounded: a bridge failure can dump a whole property list or an
        # Objective-C description, and this line lands on the press path
        # (same discipline as audio_retry's [:120] truncation).
        detail = str(e)[:120]
        print(f"CoreAudio property read failed ({detail}); treating as no evidence.")
        return None
    if status != 0 or not isinstance(raw, (bytes, bytearray)):
        return None
    return int.from_bytes(raw, sys.byteorder)


def read_live_default_input() -> LiveDefaultInput | None:
    """The live default input device, or None when CoreAudio gave no
    evidence at all (transient failure: a caller must NOT treat that as
    "unchanged" and skip a needed rebuild -- but neither is it proof of
    change, so session_needs_reset() stays quiet on it)."""
    device_id = _read_uint32(
        kAudioObjectSystemObject,
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    if device_id is None or device_id == kAudioObjectUnknown:
        # kAudioObjectUnknown is CoreAudio's "there is no default input
        # device", not a device: as a live read it would make every press
        # rebuild the session on a machine with no microphone, with nothing
        # ever settling. Absence of a device is reported like the absence of
        # an answer -- no evidence, no reset.
        return None
    alive = _read_uint32(
        device_id,
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    )
    # Doubt about liveness falls to "not alive": the safe response to an
    # unproven device is a fresh session, not a stream opened on it.
    return LiveDefaultInput(device_id=device_id, is_alive=alive == 1)


def session_needs_reset(
    live: LiveDefaultInput | None, session_device_id: int | None
) -> bool:
    """Whether the cached session was built against another device.

    A session with no recorded baseline (``session_device_id`` None) has
    never been bound, so any evidence of a live device resets it; no
    evidence at all (``live`` None) resets nothing -- the next press asks
    again."""
    if live is None:
        return False
    return (
        session_device_id is None
        or live.device_id != session_device_id
        or not live.is_alive
    )
