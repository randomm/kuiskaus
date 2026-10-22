"""The dedupe baseline must be ONE read, not two (issue #72).

Split from test_audio_device_3.py, which is at the module-cap margin. These
are the regressions for the two reset bugs:

* ``reset_session`` read CoreAudio once for its own decision
  and ``registry.open()`` read it a SECOND time for ``device_id``, while
  ``last_reset`` was pinned to the FIRST read. When the two reads
  disagreed (a coreaudiod storm is exactly when they do), the dedupe
  guard -- which never compared ``live.device_id`` to ``registry.device_id``
  -- suppressed the still-needed reset forever: the session stayed bound
  to the device the second read had named, and every later press of the
  real default was told the session was already fresh.
* the keep-warm re-arm ran BEFORE the ``if not rebuilt`` bail, so
  a rebuild that never came back re-armed a capture over a session that
  does not exist (and left no recovery ask behind for when it does).

``open()`` now records device_id and last_reset from its own single read
(and clears both when construction fails), and the reset's fallback fills
BOTH halves from its own read only when ``open()``'s read gave no
evidence -- the pair the guard compares is always one pair.
"""

from unittest.mock import MagicMock, call, patch

import kuiskaus.audio_device as device_module
from kuiskaus.audio_coreaudio import LiveDefaultInput
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input

# ---------------------------------------------------------------------------
# device_id and last_reset must come from the SAME CoreAudio read
# ---------------------------------------------------------------------------


def test_open_records_the_exact_pair_it_opened_against(coreaudio_stub):
    """Direct ``last_reset`` contract: the pair the dedupe guard compares
    against is the pair ``open()`` actually saw -- one read, written to
    both fields together, not inferred from a later or earlier one."""
    _live_default_input(coreaudio_stub, 77, is_alive=False)
    registry, _pa, _module = _registry()

    assert registry.open(_find_device) is True

    assert registry.device_id == 77
    assert registry.last_reset == LiveDefaultInput(device_id=77, is_alive=False)


def test_a_construction_failure_clears_the_stale_baseline(coreaudio_stub):
    """No instance, no baseline. A failed ``PyAudio()`` used to leave the
    PREVIOUS open's device_id/last_reset standing, so the dedupe guard
    could vouch for a session that does not exist -- and with the guard's
    pair matching the live read, no reset ever runs again to re-bind it."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    assert registry.device_id == 77
    assert registry.last_reset == LiveDefaultInput(device_id=77, is_alive=True)

    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))
    assert registry.open(_find_device) is False

    # The instance a PREVIOUS open built stays tracked and reachable (a
    # failed construction must not orphan it); what must go is the
    # BASELINE claiming this session is bound to a known device.
    assert _pa in registry._instances
    assert registry.device_id is None
    assert registry.last_reset is None


def test_split_reads_that_disagree_do_not_pin_the_dedupe_baseline(
    coreaudio_stub,
):
    """as the two independent reads produce it.

    The reset's own read says the live default is 77; ``open()``'s second
    read says 88 (a storm makes the two disagree -- exactly the moment
    this code path runs). Old code pinned ``device_id`` to 88 (open's
    read) and ``last_reset`` to 77 (the reset's read), and the guard --
    ``device_id is not None and live == last_reset``, with no
    ``live.device_id == device_id`` conjunct -- then suppressed the reset
    the session still needed on every later press at live=77. The session
    stayed bound to a device nobody is on, forever, with the user's mic
    dead until restart.
    """
    registry, stale, module = _registry()
    _live_default_input(coreaudio_stub, 66)
    registry.open(_find_device)  # baseline: device 66

    _live_default_input(coreaudio_stub, 77)  # the default moved to 77
    # open()'s second read disagrees with the reset's read, first once,
    # then (storm passed) agrees.
    open_reads = [
        LiveDefaultInput(device_id=88, is_alive=True),
        LiveDefaultInput(device_id=77, is_alive=True),
    ]
    first = MagicMock(name="PyAudioInstance-first-reset")
    second = MagicMock(name="PyAudioInstance-second-reset")
    module.PyAudio = MagicMock(side_effect=[first, second])

    with patch.object(
        device_module, "read_live_default_input", MagicMock(side_effect=open_reads)
    ):
        press_1 = registry.refresh(_find_device)
        # Press 2: live is STILL 77 while the session was pinned to 88 by
        # the split reads. The reset is still needed -- it must run.
        press_2 = registry.refresh(_find_device)

    assert press_1.busy is False
    assert press_2.busy is False
    stale.terminate.assert_called_once()
    first.terminate.assert_called_once()  # press 2 tore down press 1's session
    assert module.PyAudio.call_count == 2  # the pin bug skipped this rebuild
    assert press_2.pyaudio is second
    # And the baseline now describes the session it guards: ONE pair.
    assert registry.device_id == 77
    assert registry.last_reset == LiveDefaultInput(device_id=77, is_alive=True)


def test_a_settled_pair_dedupes_without_the_guard_pinning_a_wrong_device(
    coreaudio_stub,
):
    """The dedupe still dedupes: when both reads agree (the normal case),
    the second press must not pay a second rebuild -- the guard's new
    ``live.device_id == registry.device_id`` conjunct must not disable it."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)  # bound while alive
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77, is_alive=False)  # then it died
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio = MagicMock(side_effect=[fresh])  # tripwire: a second build fails

    first = registry.refresh(_find_device)
    second = registry.refresh(_find_device)

    assert first.busy is False and second.busy is False
    assert module.PyAudio.call_count == 1  # exactly one rebuild
    assert second.pyaudio is fresh
    assert registry.device_id == 77
    assert registry.last_reset == LiveDefaultInput(device_id=77, is_alive=False)


# ---------------------------------------------------------------------------
# re-arm only after the rebuild actually came back
# ---------------------------------------------------------------------------


def test_a_rebuild_that_never_came_back_asks_recovery_instead_of_rearming(
    coreaudio_stub,
):
    """The warm capture stopped cleanly, so ``drop()`` set no recovery flag
    -- and the old ordering called ``rearm_warm(True)`` BEFORE the
    ``if not rebuilt`` bail, arming a capture over a session that does not
    exist. Mirror the terminate_all()-bail branch instead: hand the dead
    capture back (a stopped capture is dead for presses) and ask for
    recovery so the worker path's ``recover()`` brings keep-warm back when
    the engine does."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    registry.warm.want(True)  # the menu asked for keep-warm
    _live_default_input(coreaudio_stub, 88)  # and the default moved
    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))
    warm = MagicMock(name="warm")
    warm.stop.return_value = True  # the capture stopped cleanly...
    rearm = MagicMock(name="rearm-warm")

    result = registry.refresh(_find_device, warm, rearm, None)

    assert result.pyaudio is None and result.busy is False
    assert call(True) not in rearm.call_args_list  # ...but NOT re-armed
    rearm.assert_called_once_with(False)  # dead capture handed back

    # And the session is recoverable: the worker path's recovery brings
    # keep-warm back once there is a session to arm against.
    recovery = MagicMock(name="rearm-warm-recovery")
    registry.warm.recover(recovery)
    recovery.assert_called_once_with(True)
