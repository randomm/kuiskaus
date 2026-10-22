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
# The codes below are the PyObjC bridge's own values -- the four-char
# decodings are 'dIn ' and 'livn' (they are NOT 'defi'/'aliv'). They are
# bare integers whose names are their only documentation here, so
# tests/test_audio_coreaudio.py compares all five against the framework
# itself (that test skips where pyobjc-framework-CoreAudio is absent).
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

# The bridge is an optional PyObjC framework on this machine. If importing
# it fails, that is permanent for the life of the process, and a read that
# will never work should not print once per press (or twice -- this module
# does two reads per live check). Latched by _coreaudio() on the first
# ImportError; separate from a per-read failure, which IS transient and is
# logged as one.
_coreaudio_unavailable = False

# A nonzero OSStatus is CoreAudio refusing a property, which is the
# difference between "the device is not there" and "we could not ask". It
# is bounded because the reads run on the press path: the first few per
# process are the diagnostic, the rest are the same storm.
_OSSTATUS_LOG_BUDGET = 5
_osstatus_logged = 0


def _coreaudio():
    """The CoreAudio bridge, or None when this machine has no importable one.

    A missing framework is not a property read that failed: there is nothing
    to retry, so it is latched and reported once instead of once per read.
    """
    global _coreaudio_unavailable
    if _coreaudio_unavailable:
        return None
    try:
        import CoreAudio
    except ImportError as import_error:
        _coreaudio_unavailable = True
        print(
            f"CoreAudio unavailable ({str(import_error)[:120]}); "
            "live input-device checks are off for this session."
        )
        return None
    return CoreAudio


def _reset_diagnostics_for_tests() -> None:
    """Clear the latched diagnostics (test seam; production never calls it)."""
    global _coreaudio_unavailable, _osstatus_logged
    _coreaudio_unavailable = False
    _osstatus_logged = 0


class LiveDefaultInput(NamedTuple):
    """The system default input device as CoreAudio sees it right now."""

    device_id: int
    is_alive: bool


def _read_uint32(object_id: int, selector: int, scope: int, element: int) -> int | None:
    """One UInt32 property read, or None if CoreAudio refused it.

    Three failures answer here, and they are not alike. The PyObjC bridge
    around these calls CAN raise (marshalling under a coreaudiod storm, a
    first-call import failure), and this read runs on the
    ``AudioRecorder.__init__`` and press paths, where an escaping exception
    is an app that never launches -- so nothing raises. A missing framework
    is latched and logged once; a raising call and a nonzero OSStatus are
    both transient and are each logged as what they are.
    """
    global _osstatus_logged
    CoreAudio = _coreaudio()
    if CoreAudio is None:
        return None
    try:
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
    if status != 0:
        # CoreAudio reports failure as a status code rather than raising, so
        # this is the only trace that the answer was "refused" and not "no
        # such device". Bounded for the same press-path reason as above.
        if _osstatus_logged < _OSSTATUS_LOG_BUDGET:
            _osstatus_logged += 1
            print(
                f"CoreAudio refused selector {selector} with OSStatus {status}; "
                "treating as no evidence."
            )
        return None
    if not isinstance(raw, (bytes, bytearray, memoryview)) or len(raw) != _SIZE_BYTES:
        # Not a UInt32 at all: the bridge handed back something the property
        # contract does not describe, which is no evidence rather than a
        # number to guess at.
        return None
    return int.from_bytes(bytes(raw), sys.byteorder)


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
