"""One ``Pa_Terminate`` per instance, whoever gets there first (issue #72).

``terminate_all()`` takes no recorder lock and no ``_reset_lock`` of its own:
the cleanup path (``cleanup_recorder_session``) walks straight into it while a
reset elsewhere holds ``_reset_lock`` and is mid-teardown on the same
instances. Both entrants snapshot the instance list, and both then drove the
whole teardown for everything in their snapshot. Forgetting before terminating
made the *pairing* safe; it never made the terminate unique, so on real
pyaudio two threads reach ``Pa_Terminate()`` on one instance -- which drives
PortAudio's init count below the live reference count and kills an unrelated
live instance with ``-10000`` (the mic dies on healthy hardware).

Untracking is therefore an exclusive claim: ``forget()`` answers whether THIS
call removed the instance, and the caller that did not remove it does not own
its teardown either.

The two dispositions the module owes on a refusal are tested here too: an
instance that would not terminate goes back on the list, and its index binding
comes back only if nothing else has been bound while the native call was in
flight.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from kuiskaus.audio_teardown import Teardown, dispose_claimed, dispose_prepared
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input


def _two_teardown_callers_meet_after_their_snapshots(registry, monkeypatch) -> None:
    """Force the interleaving the claim exists for.

    ``terminate_all()`` snapshots the instance list, asks the ledger gate a
    second time, and only then disposes -- so holding both threads at that
    second gate guarantees neither has forgotten anything before both have
    already copied the same list. Without the gate the second thread can
    arrive after the first has finished, which is a different (already safe)
    schedule and would let the test pass without ever exercising the race.
    """
    both_snapshotted = threading.Barrier(2, timeout=5)
    local = threading.local()
    real_gate = registry._refuse_teardown

    def gated() -> bool:
        seen = getattr(local, "calls", 0) + 1
        local.calls = seen
        if seen == 2:
            try:
                both_snapshotted.wait()
            except threading.BrokenBarrierError:
                pass  # a thread that fell behind; the count below is the verdict
        return real_gate()

    monkeypatch.setattr(registry, "_refuse_teardown", gated)


# ---------------------------------------------------------------------------
# The claim: one instance, one Pa_Terminate
# ---------------------------------------------------------------------------


def test_two_concurrent_teardowns_terminate_one_instance_exactly_once(
    coreaudio_stub, monkeypatch
):
    """Two ``terminate_all()`` callers, one registered instance, one
    ``terminate()``.

    The measured failure is the second one: both threads reach
    ``Pa_Terminate()`` on the same instance and PortAudio's reference count
    underflows, taking down a live instance neither of them named.
    """
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _two_teardown_callers_meet_after_their_snapshots(registry, monkeypatch)

    results: list[bool] = []
    results_lock = threading.Lock()

    def _teardown() -> None:
        clean = registry.terminate_all()
        with results_lock:
            results.append(clean)

    threads = [
        threading.Thread(target=_teardown, name=f"teardown-{n}", daemon=True)
        for n in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not any(t.is_alive() for t in threads), "a teardown caller wedged"
    assert live.terminate.call_count == 1, (
        f"Pa_Terminate ran {live.terminate.call_count} times on one instance: "
        "PortAudio's init count drops below the live reference count and "
        "terminates an instance neither caller named (-10000)"
    )
    assert sorted(results) == [True, True], (
        "the caller that lost the claim still has to report a clean pass -- "
        f"the instance is gone either way: {results}"
    )


def test_forget_claims_an_instance_for_exactly_one_caller(coreaudio_stub):
    """``forget()`` is the claim itself: the bool answers "did THIS call take
    it off the list", which is the only question that separates the owner of a
    teardown from a thread that walked in behind them."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)

    assert registry.forget(live) is True
    assert registry.forget(live) is False, (
        "a second caller cannot tell it lost the instance to somebody else"
    )


def test_the_caller_that_lost_the_claim_never_touches_the_native_teardown(
    coreaudio_stub,
):
    """Losing the claim is not a failure to report: the instance is gone, the
    caller that claimed it is terminating it, and this caller must not reach
    ``terminate()`` a second time."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)

    assert registry.forget(live) is True  # the other thread owns this teardown
    outcome = dispose_prepared(registry, live)
    assert outcome is Teardown.DELEGATED, (
        "losing the claim is neither a teardown this caller ran nor a refusal "
        f"it owes the budget for: {outcome}"
    )
    live.terminate.assert_not_called()


def test_the_dispose_path_the_retry_loop_uses_unbinds_before_it_terminates(
    coreaudio_stub,
):
    """``_dispose`` (``terminate()``, the attempt's own cleanup) had the old
    ordering while ``terminate_all()`` had the new one: the same torn pair --
    a terminated instance still paired with the index that resolves -- was
    readable by anything that snapshotted during the native call. Both paths
    dispose one instance, so both owe the same order."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)

    snapshots_during_terminate: list = []
    live.terminate.side_effect = lambda: snapshots_during_terminate.append(
        registry.snapshot()
    )

    registry.terminate(live)

    assert snapshots_during_terminate, "the terminate never ran"
    for instance, index in snapshots_during_terminate:
        assert index is None or instance is not live, (
            f"snapshot() paired the terminated instance with live index {index}"
        )


def test_a_refusal_from_the_retry_loop_path_keeps_the_instance_reachable(
    coreaudio_stub,
):
    """The re-register on refusal belongs to the shared ordering, not to
    ``terminate_all()`` alone: forgetting before terminating is only safe if
    every path that forgets puts the instance back when it would not die."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    live.terminate.side_effect = OSError("Pa_Terminate failed")

    registry.terminate(live)

    assert registry.tracks(live) is True, "the refused instance became unreachable"
    assert registry.snapshot() == (live, 3), (
        "the refusal lost the index that is still a fact about the live instance"
    )


# ---------------------------------------------------------------------------
# What a refusal republishes
# ---------------------------------------------------------------------------


def test_a_refusal_does_not_republish_an_index_bound_during_the_terminate(
    coreaudio_stub,
):
    """The republish is a guarded write, not a restore.

    While ``Pa_Terminate()`` was in flight a press resolved the device again
    and bound it against the instance that is still live. Republishing the
    value this teardown read -- unconditionally -- overwrites the fresher
    binding with a staler one, and every later press opens on the index
    somebody else just proved wrong.
    """
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)

    def _refuse_after_a_fresher_bind() -> None:
        registry.bind_index(9, live)  # a concurrent press re-resolved and bound
        raise OSError("Pa_Terminate failed")

    live.terminate.side_effect = _refuse_after_a_fresher_bind

    assert registry.terminate_all() is False
    assert registry.snapshot() == (live, 9), (
        "the refusal overwrote the binding a concurrent press resolved while "
        "the native call was in flight"
    )


# ---------------------------------------------------------------------------
# A pass that disposed nothing says nothing
# ---------------------------------------------------------------------------


def test_an_empty_teardown_pass_does_not_retire_the_restart_verdict(
    coreaudio_stub,
):
    """``note_terminate_outcome`` books a teardown that ran. With nothing
    registered there was no teardown, and a pass that disposed nothing must
    not report the engine clean -- that is how a verdict earned against a
    poison instance disappears on a press that never touched PortAudio."""
    registry, _live, _module = _registry()
    registry.needs_restart = True
    registry.terminate_failures = 3
    registry.pyaudio = None

    assert registry.terminate_all() is True

    assert registry.terminate_failures == 3, (
        "an empty pass booked as a clean teardown retired the refusal budget"
    )
    assert registry.needs_restart is True, (
        "an empty pass retired the verdict a real teardown earned"
    )


def test_a_pass_that_disposed_something_still_reports_clean(coreaudio_stub):
    """The guard is "did this pass dispose anything", not "did anything
    survive": one registered instance, cleanly terminated, is a clean teardown
    and still retires the verdict."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    registry.needs_restart = True
    registry.terminate_failures = 3

    assert registry.terminate_all() is True

    assert registry.terminate_failures == 0
    assert registry.needs_restart is False
    live.terminate.assert_called_once()


def test_a_lone_refusal_still_counts(coreaudio_stub):
    """...and the same guard must not swallow a refusal: one instance, refused,
    is a teardown that ran and did not come clean."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    live.terminate.side_effect = OSError("Pa_Terminate failed")

    assert registry.terminate_all() is False

    assert registry.terminate_failures == 1
    assert registry.tracks(live) is True


def test_a_half_teardown_still_names_the_instances_it_reached(coreaudio_stub, capsys):
    """The partial-teardown log line rides on the same bookkeeping, so the
    empty-pass guard has to leave it reachable for a real half teardown."""
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    stuck = MagicMock(name="PyAudioInstance-stuck")
    stuck.terminate.side_effect = OSError("Pa_Terminate failed")
    registry.register(stuck)

    assert registry.terminate_all() is False

    live.terminate.assert_called_once()
    assert stuck in registry._instances
    printed = capsys.readouterr().out
    assert "Terminated 1 of 2" in printed


# ---------------------------------------------------------------------------
# A delegation is not a clean pass
# ---------------------------------------------------------------------------


def _handed_to_a_failing_claimant(registry, instance, monkeypatch):
    """Make ``instance`` be claimed by ANOTHER teardown caller whose
    ``terminate()`` then fails, so it goes back on the registry's list.

    The window ``terminate_all()`` is exposed to: it snapshots the list, a
    second caller forgets the instance first (so this pass loses the claim and
    must not touch the native call), and that caller's refusal re-registers
    the instance -- live, tracked, and never terminated by anybody this pass
    can see. Threading cannot be scheduled deterministically here, so the
    claimant's whole teardown runs inside this pass's lost ``forget()``: the
    state the pass then resolves against is exactly the one the race produces.
    """
    real_forget = registry.forget
    handed_over = [False]

    def _forget(pa):
        if pa is instance and not handed_over[0]:
            handed_over[0] = True
            real_forget(pa)  # the claimant got there first
            instance.terminate.side_effect = OSError("Pa_Terminate failed")
            dispose_claimed(registry, pa)  # it failed, and put the instance back
            return False
        return real_forget(pa)

    monkeypatch.setattr(registry, "forget", _forget)


def test_a_delegated_instance_the_claimant_failed_keeps_the_pass_dirty(
    coreaudio_stub, monkeypatch
):
    """The claim-loss bug: a pass whose every instance belonged to somebody
    else booked ``note_terminate_outcome(clean=True)`` even when that somebody
    else FAILED, so the session retired a restart verdict it had earned and the
    next reset rebuilt over a PyAudio that never died -- PortAudio keeps the
    frozen device list and never re-enumerates, the issue #72 symptom itself.

    A delegation is only clean if the instance actually left the registry.
    """
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    registry.needs_restart = True
    registry.terminate_failures = 3
    _handed_to_a_failing_claimant(registry, live, monkeypatch)

    assert registry.terminate_all() is False, (
        "the instance is still registered: this pass terminated nothing and "
        "nobody told it so"
    )
    assert registry.tracks(live) is True
    live.terminate.assert_called_once()  # only the claimant reached the native call
    assert registry.terminate_failures == 3, (
        "a pass that terminated nothing booked a clean teardown and cleared "
        "the refusal budget"
    )
    assert registry.needs_restart is True, (
        "a pass that terminated nothing retired the verdict a real teardown "
        "earned; the next reset rebuilds over a live instance"
    )


def test_a_delegated_instance_that_reached_the_list_again_counts_as_incomplete(
    coreaudio_stub, monkeypatch, capsys
):
    """Mixed pass: one instance this caller took down, one it lost to a
    claimant that failed. The pass is incomplete, and the partial-teardown log
    says what THIS pass terminated -- not what it delegated away.
    """
    registry, live, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    delegated = MagicMock(name="PyAudioInstance-delegated")
    registry.register(delegated)
    _handed_to_a_failing_claimant(registry, delegated, monkeypatch)

    assert registry.terminate_all() is False

    live.terminate.assert_called_once()
    assert registry.tracks(delegated) is True, "the claimant's instance came back live"
    printed = capsys.readouterr().out
    assert "Terminated 1 of 2" in printed, (
        "a delegated instance is not one this pass terminated: " + printed
    )
    assert registry.terminate_failures == 1, (
        "the pass that did reach a native call still books its own attempt"
    )
