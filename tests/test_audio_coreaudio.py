"""Unit tests for kuiskaus.audio_coreaudio (issue #72).

The live CoreAudio read and the change predicate; the registry /
session-reset half stays in test_audio_device.py. No hardware and no
CoreAudio framework: the autouse ``coreaudio_stub`` fixture owns the
bridge, so a developer's real default input device can never leak into
these assertions -- with the one exception at the foot of this file, which
sets the stub aside to check the mirrored constants against the framework.
"""

import importlib
import sys

import pytest

from kuiskaus.audio_coreaudio import (
    LiveDefaultInput,
    read_live_default_input,
    session_needs_reset,
)
from tests.support_audio_recorder import _live_default_input

# 'dIn ' / 'livn' as the stubbed bridge reports them. These are the SAME
# integers the module mirrors, so they pin nothing on their own: a typo that
# kept the comment and retyped the number left the whole suite green (issue
# #72). test_constants_are_the_framework_s_own_values is the
# guard; these two stay because they are readable in the stub's plumbing.
DEFAULT_INPUT_SELECTOR = 1682533920
ALIVE_SELECTOR = 1818850926


@pytest.fixture
def real_coreaudio():
    """The genuine PyObjC CoreAudio bridge, with the autouse stub set aside.

    The stub lives in ``sys.modules`` for every test, so the real module has
    to be re-imported around it -- and the stub put back afterwards, or the
    next test would be reading this developer's microphone.
    """
    saved = sys.modules.pop("CoreAudio", None)
    try:
        try:
            return importlib.import_module("CoreAudio")
        except Exception as e:  # noqa: BLE001 - optional framework boundary
            pytest.skip(f"pyobjc-framework-CoreAudio unavailable: {e}")
    finally:
        sys.modules.pop("CoreAudio", None)
        if saved is not None:
            sys.modules["CoreAudio"] = saved


def test_constants_are_the_framework_s_own_values(real_coreaudio):
    """Every constant in this module is a
    bare integer whose name is its only documentation, and the tests mirror
    the same integers -- so a retyped number survived the suite and broke
    live detection on real hardware. This is the one place the digits are
    compared to CoreAudio itself; it skips where the framework is absent.
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
    from kuiskaus import audio_coreaudio

    def _read(object_id, address, _qs, _q, _s, _out):
        if address.mSelector == DEFAULT_INPUT_SELECTOR:
            return 0, 4, (96).to_bytes(4, audio_coreaudio.sys.byteorder)
        return 1852797029, 0, None  # '!sta' -- read failed

    coreaudio_stub.AudioObjectGetPropertyData.side_effect = _read
    assert read_live_default_input() == LiveDefaultInput(device_id=96, is_alive=False)


def test_non_bytes_property_data_is_no_evidence(coreaudio_stub):
    """A successful status with a non-bytes out-buffer is not a value."""
    from kuiskaus import audio_coreaudio

    coreaudio_stub.AudioObjectGetPropertyData.return_value = (0, 4, None)
    assert audio_coreaudio._read_uint32(1, DEFAULT_INPUT_SELECTOR, 0, 0) is None


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
