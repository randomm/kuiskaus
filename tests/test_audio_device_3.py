"""Device-index ownership and the rebuild dedupe (issue #72, review round 1).

Split from ``test_audio_device.py`` for the 500-line module cap. These are
the regressions for the two findings of the first adversarial review:

* an index is a fact about ONE PortAudio instance's frozen enumeration, so
  it must not outlive the instance it was resolved against;
* a rebuild that could not read CoreAudio a second time still has to record
  the pair it rebuilt against, or the next press rebuilds again.
"""

from unittest.mock import MagicMock, patch

import kuiskaus.audio_device as device_module
from kuiskaus.audio_coreaudio import read_live_default_input
from kuiskaus.audio_device import PyAudioRegistry
from tests.support_audio_recorder import _live_default_input


def _registry(pa_module=None, pa=None):
    """A registry over a stubbed pyaudio module yielding ``pa`` (or one
    fresh mock) per construction."""
    module = pa_module or MagicMock(name="pyaudio-module")
    if pa is None:
        pa = MagicMock(name="PyAudioInstance")
    module.PyAudio = MagicMock(return_value=pa)
    return PyAudioRegistry(module), pa, module


def _find_device(_pa):
    return 3


# ---------------------------------------------------------------------------
# An index belongs to the instance that enumerated it
# ---------------------------------------------------------------------------


def test_bound_index_refuses_an_index_from_another_instance():
    """A PortAudio index is a fact about ONE instance's frozen enumeration
    (issue #72 review, round 1): an index resolved elsewhere must never be
    served as the session's, because opening on it names a device the live
    instance never enumerated.
    """
    registry, _pa, _ = _registry()
    registry.open(_find_device)  # binds index 3 against the live instance

    registry.bind_index(9, MagicMock(name="other-instance"))

    assert registry.input_device_index == 3  # the live pairing stands


def test_a_disposed_binding_stops_naming_a_device():
    """The unbound session is where a stale index does its damage: with no
    instance to compare against, an index left behind by an attempt that
    then failed was read back forever (issue #72 review, round 1).
    """
    registry, pa, _ = _registry()
    registry.bind_index(3, pa)  # an attempt on an unbound session
    assert registry.input_device_index is None  # no live instance: not an index

    registry.pyaudio = pa  # ... and it becomes one once that instance is live
    assert registry.input_device_index == 3

    registry.terminate(pa)  # the instance goes away -- so does its index
    assert registry.input_device_index is None


def test_a_disposed_attempt_instance_forces_the_next_attempt_to_re_resolve(
    coreaudio_stub,
):
    """The wedge from the issue #72 review, round 1, end to end.

    Trace: the session is unbound, a press's attempt constructs an instance,
    resolves the default on it (old AirPods, index 3), binds that index, and
    then ``pa.open()`` fails in a coreaudiod storm -- the instance is
    disposed, the binding survives, the AirPods disconnect, and every later
    attempt trusts the pinned index, opens on a dead device, and exhausts.
    The binding is owned by the instance it was resolved on, so disposing it
    un-resolves the index and the next attempt enumerates again.
    """
    from kuiskaus.audio_retry import attempt_open_once

    module = MagicMock(name="pyaudio-module")
    registry = PyAudioRegistry(module)
    resolved: list[object] = []

    def find(pa):
        resolved.append(pa)
        return 3

    storm = MagicMock(name="pa-storm")
    storm.get_device_info_by_index.return_value = {"index": 3}
    storm.open.side_effect = OSError("-9998 storm")
    module.PyAudio = MagicMock(return_value=storm)

    _pa, stream, error, _rate = attempt_open_once(
        module, 8, 1, 16000, 1024, find, 4, 1, 0.0, registry=registry
    )

    assert stream is None and isinstance(error, OSError)
    storm.terminate.assert_called_once()  # disposed by the failed attempt
    assert registry.input_device_index is None  # ... and its binding went with it

    # The AirPods are gone; the next press must resolve afresh, not trust 3.
    revived = MagicMock(name="pa-revived")
    revived.get_device_info_by_index.return_value = {"index": 9}
    revived.open.return_value = MagicMock(name="stream")
    module.PyAudio = MagicMock(return_value=revived)

    _pa, stream, error, _rate = attempt_open_once(
        module,
        8,
        1,
        16000,
        1024,
        find,
        4,
        2,
        0.0,
        existing_pa=registry.pyaudio,
        existing_device_index=registry.input_device_index,
        registry=registry,
    )

    assert error is None and stream is not None
    assert resolved == [storm, revived]  # re-resolved on the fresh instance


# ---------------------------------------------------------------------------
# A rebuild records the pair it rebuilt against, even on thin evidence
# ---------------------------------------------------------------------------


def test_a_rebuild_records_its_pair_when_open_could_not_read_the_default(
    coreaudio_stub,
):
    """ISSUE (issue #72 review, round 1): the dedupe baseline was set from
    the refresh's own read, but ``device_id`` -- which the dedupe guard
    requires to be non-None -- came only from ``open()``'s second read. A
    CoreAudio read that failed inside ``open()`` (quite possible right
    after the storm that moved the default) left the guard unable to engage,
    so every later press paid a full teardown + rebuild on the hotkey
    callback thread: the per-press PortAudio churn the dedupe exists to stop.
    """
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77, is_alive=False)
    moved = read_live_default_input()  # the pair the reset runs against
    rebuilt = MagicMock(name="PyAudioInstance-rebuilt")
    module.PyAudio = MagicMock(return_value=rebuilt)
    # refresh()'s read succeeds; open()'s read inside the rebuild does not --
    # CoreAudio is allowed to fail transiently, and later reads recover.
    reads = [moved, None]

    def flaky_read():
        return reads.pop(0) if reads else moved

    with patch.object(
        device_module, "read_live_default_input", MagicMock(side_effect=flaky_read)
    ):
        first = registry.refresh(_find_device)
        second = registry.refresh(_find_device)

    assert first.busy is False
    assert registry.device_id == 77  # recorded from the read that DID succeed
    assert module.PyAudio.call_count == 1  # exactly ONE rebuild (the mock is
    # the one installed after open(), so its count covers the presses alone)
    assert second.pyaudio is rebuilt  # second press: deduped, nothing rebuilt
