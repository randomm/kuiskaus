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
# failed import; separate from a per-read failure, which IS transient and is
# logged as one.
_coreaudio_unavailable = False

# A nonzero OSStatus is CoreAudio refusing a property, which is the
# difference between "the device is not there" and "we could not ask". It
# is bounded because the reads run on the press path: the first few per
# process are the diagnostic, the rest are the same storm.
#
# The two diagnostics below are bounded SEPARATELY, at the same budget. They
# fail for different reasons and a reader needs both: a bridge that refuses
# five reads (a coreaudiod storm, the common shape) would otherwise spend one
# shared budget entirely on "refused" lines and leave the payload-shape line
# -- the one that explains why every reset was skipped -- permanently silent.
_OSSTATUS_LOG_BUDGET = 5
_refused_logged = 0
_malformed_logged = 0


def _coreaudio():
    """The CoreAudio bridge, or None when this machine has no importable one.

    A framework this machine cannot import is not a property read that
    failed: there is nothing to retry, so it is latched and reported once
    instead of once per read.
    """
    global _coreaudio_unavailable
    if _coreaudio_unavailable:
        return None
    try:
        import CoreAudio
    except Exception as import_error:  # noqa: BLE001 - PyObjC import boundary, logged
        # Not just ImportError: a PyObjC dylib / pyobjc-core version skew
        # raises OSError or ValueError out of the same import, and nothing
        # may raise from this read -- it runs on the AudioRecorder.__init__
        # and press paths, where an escaping exception is an app that never
        # launches (the same widening terminate_quietly() applies to
        # Pa_Terminate).
        _coreaudio_unavailable = True
        print(
            f"CoreAudio unavailable ({str(import_error)[:120]}); "
            "live input-device checks are off for this session."
        )
        return None
    return CoreAudio


def _reset_diagnostics_for_tests() -> None:
    """Clear the latched diagnostics (test seam; production never calls it)."""
    global _coreaudio_unavailable, _refused_logged, _malformed_logged
    _coreaudio_unavailable = False
    _refused_logged = 0
    _malformed_logged = 0


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
    global _refused_logged, _malformed_logged
    CoreAudio = _coreaudio()
    if CoreAudio is None:
        return None
    try:
        address = CoreAudio.AudioObjectPropertyAddress(selector, scope, element)
        status, _size, raw = CoreAudio.AudioObjectGetPropertyData(
            object_id, address, 0, _NO_QUALIFIER, _SIZE_BYTES, _OUT_BUFFER
        )
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
        if _refused_logged < _OSSTATUS_LOG_BUDGET:
            _refused_logged += 1
            print(
                f"CoreAudio refused selector {selector} with OSStatus {status}; "
                "treating as no evidence."
            )
        return None
    # ``len`` of a memoryview counts ELEMENTS, not bytes, so the check is on
    # the bytes the caller needs -- a wider element type would otherwise pass
    # a length test with the wrong amount of data behind it.
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        # Not a UInt32 at all: the bridge handed back something the property
        # contract does not describe, which is no evidence rather than a
        # number to guess at.
        bad_shape = f"a {type(raw).__name__} payload"
    elif len(payload := bytes(raw)) != _SIZE_BYTES:
        bad_shape = f"a {len(payload)}-byte payload"
    else:
        return int.from_bytes(payload, sys.byteorder)
    # The third "CoreAudio answered something unparseable" failure, and the
    # only silent one until now: a refused read logs, a raising bridge logs,
    # this just returned None -- so a bridge consistently handing back
    # garbage left no trace of why every reset was skipped. Its own budget,
    # not the refused-read one: a storm that spends those five lines must
    # still leave this one to print. Bounded either way -- this runs on the
    # press path.
    if _malformed_logged < _OSSTATUS_LOG_BUDGET:
        _malformed_logged += 1
        print(
            f"CoreAudio returned {bad_shape} for selector {selector}; "
            "treating as no evidence."
        )
    return None


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
    # Doubt about liveness (no answer at all) falls to "not alive": the safe
    # response to an unproven device is a fresh session, not a stream opened
    # on it. Any OTHER non-zero is the device saying yes -- isAlive is a raw
    # UInt32 and a driver answering something besides 1 for alive is still
    # answering alive; treating only 1 as alive rebuilt the session forever
    # over such a driver.
    return LiveDefaultInput(
        device_id=device_id, is_alive=alive is not None and alive != 0
    )


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
