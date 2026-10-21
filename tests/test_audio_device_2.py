"""The keep-warm press gate and the capture that would not stop.

Split out of test_audio_device.py, which had grown past the 500-line module
cap (issue #72 review). Everything here is the same question from the
registry's side: a keep-warm capture whose thread is parked in a native
``stream.read()`` is dropped from the recorder's state, and the session is
the only thing left that can name that thread -- so every path that would
drive PortAudio's init count to zero has to ask it first.
"""

import threading
from unittest.mock import MagicMock, call

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


def _stuck_capture():
    """A keep-warm capture whose stop() reports failure -- the thread is
    parked in a native read -- and only reports success from the third call
    on, i.e. once that read has returned and the thread exited."""
    warm = MagicMock(name="warm")
    warm.stop.side_effect = [False, False, True]
    return warm


# ---------------------------------------------------------------------------
# The keep-warm press gate
# ---------------------------------------------------------------------------


def test_refused_press_is_admitted_without_touching_the_session(
    coreaudio_stub,
):
    """The recorder refuses a press while a recording is live, so the gate
    must not reach refresh() first: that would stop the warm capture
    feeding the in-progress recording and rebuild under it (#72 review)."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)  # the default moved mid-capture
    warm = MagicMock(name="warm")
    rearm = MagicMock(name="rearm-warm")

    assert registry.press_check_if_admissible(True, warm, None, _find_device, rearm)
    warm.stop.assert_not_called()  # the live capture keeps delivering
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # nothing rebuilt
    rearm.assert_not_called()


def test_press_without_a_warm_capture_never_runs_a_press_side_reset(
    coreaudio_stub,
):
    """With no keep-warm capture there is no press-side reset: the worker
    refreshes at the top of its own loop (issue #72)."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    rearm = MagicMock(name="rearm-warm")

    assert registry.press_check_if_admissible(False, None, None, _find_device, rearm)
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1
    rearm.assert_not_called()


def test_busy_warm_reset_drops_the_capture_it_could_not_stop(coreaudio_stub):
    """A capture whose stop() timed out is dead: its stop event is set, so
    when the stuck read returns the thread exits and it can never
    re-open. The busy return must hand it back as rearm(False) so the
    recorder clears _warm -- otherwise the next press that needs no reset
    records silently into that dead sink (#72 review)."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    warm = MagicMock(name="warm")
    warm.stop.return_value = False
    rearm = MagicMock(name="rearm-warm")

    admissible = registry.press_check_if_admissible(
        False, warm, None, _find_device, rearm
    )

    assert admissible is False  # the press fails "microphone busy"
    rearm.assert_called_once_with(False)  # recorder-side drop, logged
    warm.stop.assert_called_once()
    pa.terminate.assert_not_called()  # the engine is still up -- untouched
    assert module.PyAudio.call_count == 1  # ... and nothing was rebuilt


def test_successful_warm_reset_rearms_the_capture(coreaudio_stub):
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]
    warm = MagicMock(name="warm")
    warm.stop.return_value = True
    rearm = MagicMock(name="rearm-warm")

    assert registry.press_check_if_admissible(False, warm, None, _find_device, rearm)
    warm.stop.assert_called_once()
    pa.terminate.assert_called_once()
    rearm.assert_called_once_with(True)


def test_the_worker_running_a_reset_is_never_its_own_blocker(coreaudio_stub):
    """A worker joins its predecessor and then resets: naming its own
    thread must not turn every worker-side reset into a busy report."""
    registry, stale, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio = MagicMock(return_value=fresh)

    result = registry.refresh(_find_device, None, None, threading.current_thread())

    assert result.busy is False
    stale.terminate.assert_called_once()
    assert result.pyaudio is fresh


# ---------------------------------------------------------------------------
# The capture dropped on a stuck read still holds the engine open
# ---------------------------------------------------------------------------


def test_terminate_all_refuses_while_a_dropped_capture_may_still_be_reading(
    coreaudio_stub,
):
    """CRITICAL (issue #72 review): the recorder forgets a capture it could
    not stop, so the session's handle on it is the only thing that can make
    this refusal. Without it the quit path drives PortAudio's init count to
    zero under a thread parked inside stream.read()."""
    registry, pa, _module = _registry()
    registry.open(_find_device)
    registry.warm.want(True)
    assert registry.warm.drop(_stuck_capture()) is False  # wedged in the read

    assert registry.terminate_all() is False
    pa.terminate.assert_not_called()


def test_press_does_not_terminate_under_a_capture_it_already_dropped(
    coreaudio_stub,
):
    """The press-side equivalent, and the one the prior fix could not see:
    by press 2 the recorder's ``_warm`` is None, so press 2 looks like an
    ordinary worker press with nothing warm left to fear -- yet the abandoned
    thread is still reading. Busy, and nothing rebuilt."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    registry.warm.want(True)
    stuck = _stuck_capture()
    registry.warm.drop(stuck)  # press 1 dropped it
    _live_default_input(coreaudio_stub, 77)  # and the default has moved
    rearm = MagicMock(name="rearm-warm")

    assert registry.press_check_if_admissible(False, None, None, _find_device, rearm)
    result = registry.refresh(_find_device, None, rearm, None)

    assert result.busy is True  # deferred, not half-reset
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1


def test_reset_proceeds_once_the_dropped_capture_thread_has_exited(
    coreaudio_stub,
):
    """The refusal is a gate, not a verdict: the probe releases the handle
    once the read returns, and the deferred reset then runs -- the session
    does not stay dirty for the life of the process."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    registry.warm.want(True)
    warm = MagicMock(name="warm")
    # stop(): the drop finds it wedged, and every later probe reports the
    # thread gone -- the read returned, so nothing is reading any more.
    warm.stop.side_effect = [False, True, True]
    registry.warm.drop(warm)
    _live_default_input(coreaudio_stub, 77)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]

    result = registry.refresh(_find_device, None, None, None)

    assert result.busy is False
    pa.terminate.assert_called_once()  # safe now: the reader is gone
    assert result.pyaudio is fresh


def test_terminate_all_keeps_an_instance_it_could_not_terminate(
    coreaudio_stub,
):
    """Forgetting an instance whose terminate() raised is how it becomes
    permanently unreachable while still holding PortAudio's init count above
    zero -- so it stays registered and the next attempt gets another shot
    (issue #72 review)."""
    registry, bad, _module = _registry()
    registry.open(_find_device)
    bad.terminate.side_effect = OSError("Pa_Terminate failed")
    good = MagicMock(name="PyAudioInstance-good")
    registry.register(good)

    assert registry.terminate_all() is False
    good.terminate.assert_called_once()

    assert registry.terminate_all() is False  # retried, not dropped from the list
    assert bad in registry._instances
    assert bad.terminate.call_count == 2  # the live instance is still reachable
    assert good.terminate.call_count == 1  # the terminated one was forgotten


def test_terminate_single_keeps_an_instance_it_could_not_terminate(
    coreaudio_stub,
):
    """Same rule for the single-instance path: a live instance the registry
    has forgotten is exactly the bug this class exists to prevent."""
    registry, pa, _module = _registry()
    registry.open(_find_device)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")

    registry.terminate(pa)

    assert pa in registry._instances
    pa.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# The reset that fails at terminate_all() still hands the capture back
# ---------------------------------------------------------------------------


def _terminate_refusing_registry(coreaudio_stub):
    """A bound registry over device 77 whose one instance refuses to
    terminate -- the coreaudiod-storm state terminate_quietly() and the
    terminate_all() docstring explicitly design for ("the next attempt
    gets another shot", issue #37/#60 class)."""
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    registry.warm.want(True)  # the menu asked for keep-warm
    _live_default_input(coreaudio_stub, 88)  # and the default has moved
    return registry, pa, module


def test_busy_reset_after_a_stopped_capture_hands_it_back_and_asks_recovery(
    coreaudio_stub,
):
    """CRITICAL (issue #72 review round 4): the terminate_all()-failure
    branch stopped the warm capture and then returned busy WITHOUT the
    hand-back its sibling branch does. Observed consequence: press 1
    refused, press 2 (default settled back, or one transient no-evidence
    read) admitted, and stop_recording returned the stale 7-chunk
    pre-roll as the user's speech with last_error None -- while keep-warm
    stayed off for the rest of the session with the menu still on, because
    stop() SUCCEEDED so drop() never set the recovery flag."""
    registry, pa, module = _terminate_refusing_registry(coreaudio_stub)
    warm = MagicMock(name="warm")
    warm.stop.return_value = True  # the capture stopped cleanly...
    rearm = MagicMock(name="rearm-warm")

    result = registry.refresh(_find_device, warm, rearm, None)

    assert result.busy is True
    rearm.assert_called_once_with(False)  # ...but is dead: handed back
    pa.terminate.assert_called_once()  # the refusal came from terminate
    assert module.PyAudio.call_count == 1  # busy: nothing rebuilt


def test_recovery_flag_from_a_bailed_reset_brings_keep_warm_back(
    coreaudio_stub,
):
    """The recovery half of the same finding: with only the rearm(False)
    hand-back, _recover_pending was still False (drop() sets it only on a
    FAILED stop), so WarmKeep.recover() -- reachable from the worker path
    -- never fired and keep-warm never returned. The bailed reset must
    leave the recorder able to come back via recover()."""
    registry, _pa, _module = _terminate_refusing_registry(coreaudio_stub)
    warm = MagicMock(name="warm")
    warm.stop.return_value = True
    rearm = MagicMock(name="rearm-warm")

    registry.refresh(_find_device, warm, rearm, None)
    registry.warm.recover(rearm)  # the worker path's recovery call

    assert rearm.call_args_list == [call(False), call(True)]
    # A second recover() with nothing pending must not re-arm again.
    registry.warm.recover(rearm)
    assert rearm.call_count == 2


# ---------------------------------------------------------------------------
# A persistently not-alive default costs one rebuild, not one per press
# ---------------------------------------------------------------------------


def test_persistent_dead_default_rebuilds_once_not_every_press(
    coreaudio_stub,
):
    """ISSUE (issue #72 review): refresh() re-records device_id but not
    the liveness it reset against, so a stable default the alive-read
    reports dead (or fails to read) re-triggered a full teardown + rebuild
    on EVERY press forever -- measured at one PyAudio() construction per
    refresh, each one Pa_Terminate-to-zero + Pa_Initialize + enumerate,
    on the hotkey callback thread in the keep-warm path. A second read of
    the same (device, dead) pair after a successful reset is nothing new.
    """
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)  # bound while alive
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77, is_alive=False)  # then it died
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]  # tripwire: a second build fails

    first = registry.refresh(_find_device)
    second = registry.refresh(_find_device)

    assert first.busy is False
    assert module.PyAudio.call_count == 2  # open() + exactly ONE rebuild
    assert second.busy is False
    assert second.pyaudio is fresh  # the rebuilt session is the answer
    fresh.terminate.assert_not_called()  # ... and it is not torn down again


def test_a_moved_default_after_a_dead_reset_still_resets(coreaudio_stub):
    """The dedupe must pin only the pair already reset against: a moved
    default is a different pair, even at the same liveness."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77, is_alive=False)
    registry.refresh(_find_device)
    assert module.PyAudio.call_count == 2

    _live_default_input(coreaudio_stub, 88, is_alive=False)
    result = registry.refresh(_find_device)

    assert result.busy is False
    assert module.PyAudio.call_count == 3  # pair changed: reset ran


def test_a_failed_reset_is_retried_next_press_not_deduped(coreaudio_stub):
    """The dedupe records only FULLY successful resets -- a rebuild whose
    construction failed leaves nothing recorded, and the next press must
    take another shot rather than be told the session is already fresh."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77, is_alive=False)
    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))

    failed = registry.refresh(_find_device)
    assert failed.pyaudio is None  # session left unbound

    good = MagicMock(name="PyAudioInstance-good")
    module.PyAudio = MagicMock(return_value=good)
    retried = registry.refresh(_find_device)

    assert retried.pyaudio is good  # retried, not deduped
    assert retried.busy is False
