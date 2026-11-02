"""The new-evidence exemption: who may buy a rebuild, and who may not.

The teardown backoff exists to spare the press a rebuild that is known
futile, so a device that moved by hand has to be able to buy its way out of
the window. That allowance is finite (``MAX_NEW_EVIDENCE_EXEMPTIONS``) and
buys the REBUILD -- which makes every bail before the rebuild a press that
must not pay for one: the ledger gate, a keep-warm capture that would not
stop, a ``terminate_all()`` that refused. Each of those used to sit below
the claim, so presses that never rebuilt drained the allowance and left a
genuinely hand-moved default refused for the rest of the window.

The shape the allowance is hardest to earn against is the AirPod one: a
default that alternates A,B,A,B on every read. There the same-device streak
restarts on every press, so neither churn guard fires and every press pays a
full ``Pa_Terminate``-to-zero pass -- bounded here by counting consecutive
rebuilds whatever device they name, and by wanting a device the session has
not rebuilt against rather than the counted pair bouncing back.
"""

import threading
from itertools import cycle
from unittest.mock import MagicMock, patch

import kuiskaus.audio_device as device_module
import kuiskaus.audio_device_reset as reset_module
from kuiskaus.audio_coreaudio import LiveDefaultInput
from kuiskaus.audio_device_reset import MICROPHONE_NOT_SETTLING
from kuiskaus.audio_rebuild_budget import MAX_NEW_EVIDENCE_EXEMPTIONS
from tests.support_audio_device import (
    _device_verdict_backoff_open,
    _find_device,
    _registry,
)
from tests.support_audio_recorder import _live_default_input


def test_an_alternating_default_is_bounded_and_the_backoff_holds(
    coreaudio_stub, monkeypatch
):
    """Both churn guards were keyed on ONE device, and both missed the
    canonical AirPod shape. With the default reported A,B,A,B... on each
    press (open()'s own read transiently giving no evidence, so the reset's
    fallback records the press's pair -- the shape that makes every rebuild
    target a DIFFERENT device), the same-device streak restarts on every
    press and never fires: measured, 30 presses cost 30 full
    ``Pa_Terminate``-to-zero passes and 31 ``Pa_Initialize`` constructions,
    ``needs_restart`` False, no verdict ever surfaced -- and with a verdict
    on the record, presses inside the still-open window keep paying the
    futile teardown, because "different from the last device counted" is
    true forever for a pair bouncing back. The budget now also counts
    consecutive rebuilds whatever device they name, and the exemption wants
    a device the session has not rebuilt against during this run -- not the
    counted pair bouncing.
    """
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    presses = cycle([LiveDefaultInput(88, True), LiveDefaultInput(77, True)])

    with (
        patch.object(
            reset_module,
            "read_live_default_input",
            MagicMock(side_effect=presses),
        ),
        # open()'s own read transiently gives no evidence -- quite possible
        # right after the storm that moved the default -- so the reset's
        # fallback records the pair THIS read saw, and the next press
        # legitimately sees the OTHER device.
        patch.object(
            device_module, "read_live_default_input", MagicMock(return_value=None)
        ),
    ):
        for _ in range(reset_module.MAX_RESET_REBUILDS * 10):
            registry.refresh(_find_device)

        assert registry.needs_restart is True, (
            "the alternating default never earned a verdict: the budget is "
            "still keyed on one device"
        )
        assert registry.terminate_failures == 0  # every teardown released
        assert registry.press_refusal_reason() == MICROPHONE_NOT_SETTLING
        builds = module.PyAudio.call_count
        assert builds <= reset_module.MAX_RESET_REBUILDS + 2, (
            f"the alternating default churned PortAudio {builds} times"
        )

        # ... and inside the still-open window the flap costs nothing more:
        # the exemption is for a moved default, not for the counted pair.
        for _ in range(10):
            registry.refresh(_find_device)
        assert module.PyAudio.call_count == builds, (
            "the backoff honoured a flapping default as new evidence on "
            "every press: the futile teardown still runs inside the window"
        )


def test_a_ledger_deferred_press_does_not_spend_the_exemption(
    coreaudio_stub, monkeypatch
):
    """The new-evidence exemption buys the REBUILD, so only a press that
    reaches the rebuild may pay for it. The claim used to sit inside
    ``_press_is_new_evidence`` -- a predicate -- so a press the ledger gate
    then deferred (a worker still inside a native call) spent an exemption
    on a teardown that never ran: deferred presses alone could drain the
    whole allowance and leave a later hand-moved default refused for the
    rest of the window.
    """
    registry, _pa, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _device_verdict_backoff_open(registry, monkeypatch)

    # The default moves to a device the budget has never counted: new
    # evidence, and the press wants the teardown -- but a live worker is
    # still inside a native call.
    _live_default_input(coreaudio_stub, 99, is_alive=True)
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    try:
        deferred = registry.refresh(_find_device, None, None, lambda: (worker,))
    finally:
        release.set()
        worker.join(timeout=5.0)

    assert deferred.busy is True
    assert registry.rebuild.new_evidence_spent == 0, (
        "the ledger-deferred press paid an exemption for a rebuild it never ran"
    )

    # The press that does reach the rebuild spends it.
    result = registry.refresh(_find_device)
    assert result.busy is False
    assert registry.rebuild.new_evidence_spent == 1


def test_a_stuck_keep_warm_capture_does_not_spend_the_exemption(
    coreaudio_stub, monkeypatch
):
    """The same rule one gate later: a press whose keep-warm capture would
    not stop returns busy with NO teardown and no rebuild, so it may not
    pay for one either. The claim sat above the ``warm.drop()`` bail, so
    eight such presses inside one window could drain the whole allowance
    and leave a genuinely hand-moved default refused for the rest of it.
    """
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _device_verdict_backoff_open(registry, monkeypatch)
    _live_default_input(coreaudio_stub, 99, is_alive=True)

    capture = MagicMock(name="WarmCapture-stuck")
    capture.stop.return_value = False  # parked in a native read
    registry.warm.note_installed(capture)

    result = registry.refresh(_find_device, capture, lambda _usable: None)

    assert result.busy is True
    assert registry.rebuild.new_evidence_spent == 0, (
        "the drop-deferred press paid an exemption for a rebuild it never ran"
    )
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # only the setup open()


def test_a_refused_teardown_does_not_spend_the_exemption(coreaudio_stub, monkeypatch):
    """And one gate later still: ``terminate_all()`` refusing (an instance
    that would not terminate) is the other deferral bail that sat BELOW the
    claim, so the press paid for a rebuild that never happened. The
    exemption belongs to the press that actually rebuilds.
    """
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _device_verdict_backoff_open(registry, monkeypatch)
    _live_default_input(coreaudio_stub, 99, is_alive=True)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")

    result = registry.refresh(_find_device)

    assert result.busy is True
    assert pa.terminate.call_count == 1  # the teardown was ATTEMPTED
    assert registry.rebuild.new_evidence_spent == 0, (
        "the teardown-deferred press paid an exemption for a rebuild it never ran"
    )
    assert module.PyAudio.call_count == 1  # no rebuild happened


def test_a_spent_allowance_is_refused_before_the_teardown_runs(
    coreaudio_stub, monkeypatch
):
    """The bound stays a bound on the TEARDOWN, not just on the rebuild.
    Moving the claim to the point of no return must not turn the check into
    a post-teardown one: with the allowance already spent, an unseen device
    inside the window is refused before a single instance is terminated --
    otherwise every such press pays the full stall the allowance exists to
    bound. Also pins the pure/impure split: ``exemption_available`` gates
    early, ``claim_exemption`` spends late, exactly once per rebuild."""
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _device_verdict_backoff_open(registry, monkeypatch)
    registry.rebuild.new_evidence_spent = MAX_NEW_EVIDENCE_EXEMPTIONS
    _live_default_input(coreaudio_stub, 99, is_alive=True)

    result = registry.refresh(_find_device)

    assert result.busy is True
    assert pa.terminate.call_count == 0, (
        "a press past the allowance still paid the futile teardown"
    )
    assert module.PyAudio.call_count == 1
    assert registry.rebuild.new_evidence_spent == MAX_NEW_EVIDENCE_EXEMPTIONS


def test_a_rebuild_outside_the_backoff_does_not_touch_the_allowance(
    coreaudio_stub, monkeypatch
):
    """The late claim sits under the captured ``backoff`` flag for a
    reason: an ordinary rebuild -- no verdict, window closed -- is the
    normal hand-switch path issue #72 exists to serve, and it must not
    consume a single exemption on its way by."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    monkeypatch.setattr(reset_module, "_now", lambda: 0.0)
    _live_default_input(coreaudio_stub, 99, is_alive=True)

    result = registry.refresh(_find_device)

    assert result.busy is False
    assert registry.device_id == 99
    assert registry.rebuild.new_evidence_spent == 0
    assert module.PyAudio.call_count == 2  # setup open + the rebuild
