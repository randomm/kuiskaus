"""The rebuild budget counts the session's churn, not one device's.

Two holes in ``RebuildBudget``, both the same shape -- a counter that a
device the session has never seen could reset:

* ``note_rebuild`` restarted ``count`` at 1 for any device id not already
  in ``recent``, so a default reported as a *different* device on every
  press (the ids a re-joining AirPod presents) never accumulated, never
  reached ``MAX_RESET_REBUILDS``, and the session paid a full teardown and
  ``Pa_Initialize`` on every press with no verdict on the record;
* ``recent`` grew for every id ever counted, so the same run kept the
  memory of every device it had seen, unbounded, for the life of the
  process.

The bound is now ``count`` since the last ``settle()``, whatever device the
rebuilds name; novelty is used only for the backoff's new-evidence
exemption, and ``recent`` holds the last few counted ids.
"""

from __future__ import annotations

from kuiskaus.audio_rebuild_budget import MAX_RESET_REBUILDS, RebuildBudget
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input


def test_a_run_of_new_device_ids_still_trips_the_bound(coreaudio_stub, monkeypatch):
    """The counter the bound reads must not restart on an unseen id.

    Each press reports a device the session has never rebuilt against --
    the shape the pre-fix ``note_rebuild`` treated as progress every time.
    """
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    clock = [0.0]
    monkeypatch.setattr("kuiskaus.audio_device_reset._now", lambda: clock[0])

    for n in range(MAX_RESET_REBUILDS):
        _live_default_input(coreaudio_stub, 200 + n)
        registry.refresh(_find_device)

    assert registry.rebuild.count == MAX_RESET_REBUILDS
    assert registry.needs_restart is True, (
        "a stream of new device ids never earned a verdict: the bound "
        "restarts on an unseen id"
    )
    assert module.PyAudio.call_count == MAX_RESET_REBUILDS + 1


def test_the_counted_ids_are_kept_bounded():
    """The counted-device list is a window, not a ledger for the run."""
    budget = RebuildBudget()
    for n in range(MAX_RESET_REBUILDS + 20):
        budget.note_rebuild(n)

    assert budget.count == MAX_RESET_REBUILDS + 20  # bounded by the caller
    assert len(budget.recent) <= 8
    assert budget.is_new_device(MAX_RESET_REBUILDS + 19) is False  # still counted
    assert budget.is_new_device(0) is True  # forgotten by the window


def test_the_streak_still_restarts_for_a_hand_moved_device():
    """The companion rule, unchanged: the streak is the same-device claim,
    and a switch by hand must not inherit a dead device's streak (the
    verdict names the device)."""
    budget = RebuildBudget()
    budget.note_rebuild(77)
    budget.note_rebuild(77)
    budget.note_rebuild(88)

    assert budget.streak == 1
    assert budget.device == 88
    assert budget.count == 3  # the cross-device bound does not restart
