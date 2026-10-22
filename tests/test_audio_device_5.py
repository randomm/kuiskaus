"""A PortAudio that will not come down has to say so (issue #72).

``terminate_all()`` keeps an instance it could not terminate registered --
forgetting a live instance is how it becomes permanently unreachable -- and
that is correct: retrying is the only way back. What the design did not have
was a give-up-and-surface path. Measured against the real registry with one
instance whose ``terminate()`` keeps failing and the live default moved:

    press 1: busy=True live_is_stale=True constructions=1 terminate_attempts=1
    press 2: busy=True live_is_stale=True constructions=1 terminate_attempts=2
    ...      (forever)

The reset reported busy on every press, never rebuilt, and left
``registry.device_id`` pinned to the dead device: "recording is broken until
restart", the exact state this issue exists to eliminate -- with ``last_error``
never set and the menubar still showing Ready. The trigger is reachable
through the installed pyaudio: ``PyAudio.terminate()`` force-closes
``self._streams`` and only drops a stream from that list after a successful
close, and ``close_stream_quietly`` swallows the OSError from a failed close,
so one stream that would not close leaves every later ``terminate()`` raising.

So the refusals are now counted, and past a budget of consecutive failed
teardowns the session says the one true thing: this microphone needs the app
restarted. The counter counts *teardown* refusals only -- a gate refusal from a
thread that may still be native is self-limiting and must never be reported as
a dead microphone.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import kuiskaus.audio_device_reset as reset_module
from kuiskaus.audio_device_reset import MAX_TERMINATE_FAILURES
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input


def _poison_registry(coreaudio_stub):
    """A bound registry whose one instance will never terminate, with the
    live default moved off the device it was built against.

    Every press therefore wants the reset, runs the teardown, and is told
    busy -- the measured loop, reproduced rather than theorised.
    """
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    _live_default_input(coreaudio_stub, 88)
    return registry, pa, module


def _fake_clock(monkeypatch):
    """Pin the reset's retry-backoff clock so a test can advance it."""
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    return clock


def _live_worker() -> tuple[threading.Event, threading.Thread]:
    hold = threading.Event()
    worker = threading.Thread(target=hold.wait, daemon=True)
    worker.start()
    return hold, worker


# ---------------------------------------------------------------------------
# The budget, and the verdict it produces
# ---------------------------------------------------------------------------


def test_the_early_refusals_stay_quiet_because_a_retry_may_still_succeed(
    coreaudio_stub,
):
    """Retrying is correct -- a transient coreaudiod storm clears -- so a
    single failed teardown must not startle the user with "restart the app".
    """
    registry, _pa, _module = _poison_registry(coreaudio_stub)

    for _ in range(MAX_TERMINATE_FAILURES - 1):
        assert registry.refresh(_find_device).busy is True
        assert registry.needs_restart is False


def test_the_reset_keeps_trying_and_never_rebuilds_while_the_instance_stays_up(
    coreaudio_stub, monkeypatch
):
    """One terminate attempt and zero constructions, bounded by the retry
    backoff, with the session pointing at the instance nobody can tear down.
    Nothing here abandons the teardown for good -- only the *silence* about it
    is what the budget ends, and the backoff is what stops a known-futile
    teardown stalling every press."""
    registry, pa, module = _poison_registry(coreaudio_stub)
    clock = _fake_clock(monkeypatch)
    presses = MAX_TERMINATE_FAILURES * 2

    results = [registry.refresh(_find_device) for _ in range(presses)]

    assert all(r.busy for r in results)
    assert pa.terminate.call_count == MAX_TERMINATE_FAILURES  # budget, then backoff
    assert module.PyAudio.call_count == 1  # ... and never rebuilt
    assert registry.pyaudio is pa  # still the instance that will not die

    # The backoff expires and the teardown is attempted again: deferred,
    # never abandoned.
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    registry.refresh(_find_device)
    assert pa.terminate.call_count == MAX_TERMINATE_FAILURES + 1


def test_a_teardown_that_never_succeeds_ends_in_a_surfaced_restart_need(
    coreaudio_stub,
):
    """Past the budget the session has to admit it. The user
    gets a dead microphone either way; the only question is whether anything
    says so."""
    registry, _pa, _module = _poison_registry(coreaudio_stub)

    for _ in range(MAX_TERMINATE_FAILURES):
        assert registry.refresh(_find_device).busy is True

    assert registry.needs_restart is True


# ---------------------------------------------------------------------------
# What does NOT count
# ---------------------------------------------------------------------------


def test_a_gate_refusal_from_a_live_thread_never_costs_the_restart_verdict(
    coreaudio_stub, monkeypatch
):
    """A thread that may still be inside a native call is a deferral, not a
    dead microphone: that gate releases on its own once the thread leaves
    PortAudio (the same "a gate, not a verdict" rule WarmKeep follows).
    Reporting it as "needs an app restart" would send the user to quit the app
    over a wedged read that was about to return."""
    registry, pa, _module = _poison_registry(coreaudio_stub)
    _fake_clock(monkeypatch)
    hold, worker = _live_worker()
    registry.note_worker(worker)
    try:
        for _ in range(MAX_TERMINATE_FAILURES * 2):
            assert registry.refresh(_find_device).busy is True
        assert registry.needs_restart is False
        pa.terminate.assert_not_called()  # the teardown never even ran
    finally:
        hold.set()
        worker.join(timeout=5.0)


def test_a_half_teardown_says_which_instances_it_reached(coreaudio_stub, capsys):
    """terminate_all() can terminate the live instance and still report busy
    -- a second registered instance refused. The session must not be left
    naming the instance this call DID terminate (a press would then open on a
    terminated instance and report "microphone unavailable" on healthy
    hardware), so it is unbound with it. The refusal still counts, and the
    instance that survived stays reachable.
    """
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    stuck = MagicMock(name="PyAudioInstance-stuck")
    stuck.terminate.side_effect = OSError("Pa_Terminate failed")
    registry.register(stuck)
    _live_default_input(coreaudio_stub, 88)

    assert registry.refresh(_find_device).busy is True

    live.terminate.assert_called_once()  # the live session really did go
    assert stuck in registry._instances  # ... and the poison one stays findable
    printed = capsys.readouterr().out
    assert "Terminated 1 of 2" in printed


def test_a_clean_teardown_clears_the_restart_verdict(coreaudio_stub, monkeypatch):
    """The verdict describes the teardown, so it cannot outlive a clean one:
    once the poison instance is gone the engine is healthy again and a later
    storm must start the budget from scratch rather than re-trip on sight."""
    registry, pa, module = _poison_registry(coreaudio_stub)
    clock = _fake_clock(monkeypatch)
    for _ in range(MAX_TERMINATE_FAILURES):
        registry.refresh(_find_device)
    assert registry.needs_restart is True

    pa.terminate.side_effect = None  # the storm cleared
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio = MagicMock(return_value=fresh)
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0  # the backoff expires
    result = registry.refresh(_find_device)

    assert result.busy is False
    assert result.pyaudio is fresh
    assert registry.needs_restart is False

    # ... and the budget really did restart: moving the default again and
    # failing one short of the threshold must claim nothing yet. Without the
    # reset-on-success, this is the press that would re-trip instantly.
    _live_default_input(coreaudio_stub, 99)
    fresh.terminate.side_effect = OSError("Pa_Terminate failed")
    for _ in range(MAX_TERMINATE_FAILURES - 1):
        assert registry.refresh(_find_device).busy is True
    assert registry.needs_restart is False
