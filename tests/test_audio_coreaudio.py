"""Unit tests for kuiskaus.audio_coreaudio (issue #72).

The live CoreAudio read and the change predicate; the registry /
session-reset half stays in test_audio_device.py. No hardware and no
CoreAudio framework: the autouse ``coreaudio_stub`` fixture owns the
bridge, so a developer's real default input device can never leak into
these assertions -- with the one exception at the foot of this file, which
compares the mirrored constants against the framework itself (imported at
collection time, above, so the stub cannot make the guard skip).
"""

import sys

import pytest

from kuiskaus.audio_coreaudio import (
    LiveDefaultInput,
    read_live_default_input,
    session_needs_reset,
)
from tests.support_audio_recorder import _live_default_input

# The genuine PyObjC bridge, captured at COLLECTION time -- before the
# autouse ``coreaudio_stub`` fixture takes ``sys.modules["CoreAudio"]``.
# The old fixture popped the stub and re-imported, but re-importing
# re-runs PyObjC's bridge registration, which raises once the framework
# has been loaded by anything else in the run (AVFoundation imports it in
# tests/test_permissions.py) -- so the guard skipped SILENTLY in a
# full-suite run, and a typo'd constant would have shipped green. Import
# once here (where the stub is not yet installed) and hand the module
# object to the fixture; the skip happens at collection, where skipping
# is loud, rather than mid-run.
try:
    import CoreAudio as _real_coreaudio
except Exception:  # noqa: BLE001 - optional framework boundary
    _real_coreaudio = None

# 'dIn ' / 'livn' as the stubbed bridge reports them. These are the SAME
# integers the module mirrors, so they pin nothing on their own: a typo that
# kept the comment and retyped the number left the whole suite green (issue
# #72). test_constants_are_the_framework_s_own_values is the
# guard; these two stay because they are readable in the stub's plumbing.
DEFAULT_INPUT_SELECTOR = 1682533920
ALIVE_SELECTOR = 1818850926


@pytest.fixture
def real_coreaudio():
    """The genuine PyObjC CoreAudio bridge (see ``_real_coreaudio``).

    The stub owns ``sys.modules["CoreAudio"]`` during the test, so the
    module object must come from the collection-time import -- popping the
    stub and re-importing would re-run the PyObjC bridge registration and
    raise (see the comment above), silently skipping the guard.
    """
    if _real_coreaudio is None:
        pytest.skip("pyobjc-framework-CoreAudio unavailable")
    return _real_coreaudio


def test_constants_are_the_framework_s_own_values(real_coreaudio):
    """Every constant in this module is a
    bare integer whose name is its only documentation, and the tests mirror
    the same integers -- so a retyped number survived the suite and broke
    live detection on real hardware. This is the one place the digits are
    compared to CoreAudio itself; it skips where the framework is absent
    (loudly, at collection).
    """
    from kuiskaus import audio_coreaudio

    for name in (
        "kAudioObjectSystemObject",
        "kAudioHardwarePropertyDefaultInputDevice",
        "kAudioDevicePropertyDeviceIsAlive",
        "kAudioObjectPropertyScopeGlobal",
        "kAudioObjectPropertyElementMain",
    ):
        assert getattr(audio_coreaudio, name) == getattr(real_coreaudio, name), (
            f"{name} is not what CoreAudio reports"
        )


# ---------------------------------------------------------------------------
# Live CoreAudio read
# ---------------------------------------------------------------------------


def test_no_evidence_from_coreaudio_reads_as_none(coreaudio_stub):
    # The stub's default is "every read fails" -- a nonzero OSStatus is
    # CoreAudio's way of saying "no answer", which must not become
    # "the device did not change".
    assert read_live_default_input() is None


def test_live_read_returns_default_id_and_liveness(coreaudio_stub):
    _live_default_input(coreaudio_stub, 96)
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=True)


def test_live_read_reports_dead_device(coreaudio_stub):
    _live_default_input(coreaudio_stub, 96, is_alive=False)
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=False)


def test_failed_liveness_read_doubts_the_device(coreaudio_stub):
    """A device whose liveness cannot be proven is treated as NOT alive:
    the safe response to an unproven device is a rebuild, not a stream."""

    def _read(object_id, address, _qs, _q, _s, _out):
        if address.mSelector == DEFAULT_INPUT_SELECTOR:
            return 0, 4, (96).to_bytes(4, sys.byteorder)
        return 1852797029, 0, None  # '!sta' -- read failed

    coreaudio_stub.AudioObjectGetPropertyData.side_effect = _read
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=False)


def test_non_bytes_property_data_is_no_evidence(coreaudio_stub):
    """A successful status with a non-bytes out-buffer is not a value."""
    from kuiskaus import audio_coreaudio

    coreaudio_stub.AudioObjectGetPropertyData.return_value = (0, 4, None)
    assert audio_coreaudio._read_uint32(1, DEFAULT_INPUT_SELECTOR, 0, 0) is None


def test_a_wrong_shape_payload_logs_a_bounded_diagnostic(coreaudio_stub, capsys):
    """A wrong-type or wrong-length payload is the third
    "CoreAudio answered something unparseable" failure -- and it was the
    only silent one: a refused read logs, a raising bridge logs, this just
    returned None, so a bridge that consistently handed back garbage left
    no trace of WHY every reset was skipped. One line per read, under the
    same log budget as the refused-read line (this runs on the press path).
    """
    from kuiskaus import audio_coreaudio

    wrong_type = (0, 4, 42)  # a successful status carrying an int
    wrong_length = (0, 4, b"\x60\x00")  # two bytes, not a UInt32
    for payload in (wrong_type, wrong_length) * (audio_coreaudio._OSSTATUS_LOG_BUDGET):
        coreaudio_stub.AudioObjectGetPropertyData.return_value = payload
        assert audio_coreaudio._read_uint32(1, DEFAULT_INPUT_SELECTOR, 0, 0) is None

    printed = capsys.readouterr().out
    assert "CoreAudio returned" in printed
    assert (
        printed.count("CoreAudio returned") <= audio_coreaudio._OSSTATUS_LOG_BUDGET
    ), "the payload-shape diagnostic prints unbounded on the press path"
    assert len(printed) < 1000, f"unbounded payload diagnostic: {len(printed)} chars"


def test_a_memoryview_is_measured_in_bytes_not_elements(coreaudio_stub):
    """``len`` of a memoryview counts ELEMENTS, so a 2-element view of 2-byte
    items is 4 long and 2 bytes short of a UInt32. Reading it as a device id
    would invent one out of a truncated buffer, so the check has to be on the
    byte length."""
    from kuiskaus import audio_coreaudio

    short = memoryview(bytearray(b"\x60\x00\x00\x00" + b"\xff" * 28)).cast("Q")
    assert len(short) == 4  # what a naive length check would accept
    coreaudio_stub.AudioObjectGetPropertyData.return_value = (0, 4, short)
    assert audio_coreaudio._read_uint32(1, DEFAULT_INPUT_SELECTOR, 0, 0) is None

    whole = memoryview(bytearray((96).to_bytes(4, sys.byteorder)))
    coreaudio_stub.AudioObjectGetPropertyData.return_value = (0, 4, whole)
    assert audio_coreaudio._read_uint32(1, DEFAULT_INPUT_SELECTOR, 0, 0) == 96


def test_no_default_device_at_all_is_no_evidence(coreaudio_stub):
    """kAudioObjectUnknown (0) is CoreAudio's "there is no default input
    device", not a device. As a live read it would rebuild the session on
    every press of a machine with no microphone, and nothing would ever
    settle -- so absence of a device answers like absence of a value."""
    _live_default_input(coreaudio_stub, 0)
    live = read_live_default_input()
    assert live is None
    assert session_needs_reset(live, None) is False


def test_a_raising_bridge_read_is_no_evidence(coreaudio_stub, capsys):
    """The PyObjC bridge itself can raise (marshalling under a coreaudiod
    storm, a first-call import failure). The reader's contract is None --
    'no evidence' -- never an exception: this read runs on the
    AudioRecorder.__init__ path (menubar/app construct it unguarded) and
    on the press path, so an escape there is an app that will not launch
    or a hotkey handler that dies mid-press (#72)."""
    coreaudio_stub.AudioObjectGetPropertyData.side_effect = OSError("coreaudiod storm")
    assert read_live_default_input() is None
    assert "CoreAudio property read failed" in capsys.readouterr().out


def test_an_import_that_raises_beyond_importerror_is_no_evidence_not_a_crash(
    coreaudio_stub, monkeypatch, capsys
):
    """The module's own contract is that nothing raises out of this read -- it
    runs on the ``AudioRecorder.__init__`` path (app.py and menubar.py
    construct it unguarded) and on the press path, where an escape is an app
    that never launches. A PyObjC dylib or pyobjc-core version skew raises
    ``OSError``/``ValueError`` from the import itself, not ``ImportError``,
    and the guard caught only the latter: reproduced, the OSError propagated
    out of ``registry.open()`` and up to app launch."""
    from kuiskaus import audio_coreaudio

    class _SkewFinder:
        def find_spec(self, name, path=None, target=None):
            if name == "CoreAudio":
                raise OSError("dlopen(): PyObjC bridge bundle mismatch")

    monkeypatch.delitem(sys.modules, "CoreAudio")
    monkeypatch.setattr(sys, "meta_path", [_SkewFinder(), *sys.meta_path])

    assert read_live_default_input() is None
    assert "CoreAudio unavailable" in capsys.readouterr().out

    # Latched like a missing framework: the verdict is permanent for the
    # process, so a later read neither re-raises nor re-prints.
    assert read_live_default_input() is None
    assert capsys.readouterr().out == ""
    assert audio_coreaudio._coreaudio_unavailable is True


def test_a_raising_address_constructor_is_no_evidence(coreaudio_stub):
    coreaudio_stub.AudioObjectPropertyAddress.side_effect = RuntimeError(
        "bridge marshalling"
    )
    assert read_live_default_input() is None


def test_a_raising_bridge_logs_a_bounded_message(coreaudio_stub, capsys):
    """The bridge can raise carrying an Objective-C description or a whole
    property list, and this line prints on the press path -- so the logged
    text is bounded, like audio_retry bounds its own boundary text."""
    coreaudio_stub.AudioObjectGetPropertyData.side_effect = OSError("x" * 5000)
    assert read_live_default_input() is None
    printed = capsys.readouterr().out
    assert "CoreAudio property read failed" in printed
    assert len(printed) < 300, f"unbounded bridge text on the press path: {printed}"


# ---------------------------------------------------------------------------
# The change predicate
# ---------------------------------------------------------------------------


def test_no_evidence_never_resets():
    assert session_needs_reset(None, 42) is False
    assert session_needs_reset(None, None) is False


def test_unbound_session_binds_on_first_evidence():
    assert session_needs_reset(LiveDefaultInput(7, True), None) is True


def test_same_alive_device_needs_no_reset():
    assert session_needs_reset(LiveDefaultInput(7, True), 7) is False


def test_moved_default_or_dead_device_resets():
    assert session_needs_reset(LiveDefaultInput(8, True), 7) is True
    assert session_needs_reset(LiveDefaultInput(7, False), 7) is True
