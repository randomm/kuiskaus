"""A device that will not settle must not cost a rebuild on every press.

The reset's dedupe guard compares the live CoreAudio pair against the pair
the session was last rebuilt against. When CoreAudio is steady that is
exactly one rebuild per device move. It is NOT exactly one when the pair
itself oscillates -- an AirPod stream whose ``isAlive`` flag flips between
reads through a coreaudiod storm -- because each press then legitimately
sees a different pair, the guard cannot engage, and each press pays a full
terminate-all + ``Pa_Initialize`` + re-enumeration on the hotkey callback
thread. Nothing bounded that, so the "one rebuild" property issue #72 is
about was only ever true for a well-behaved CoreAudio.
"""

import threading
from itertools import cycle
from unittest.mock import MagicMock, patch

import kuiskaus.audio_device as device_module
import kuiskaus.audio_device_reset as reset_module
from kuiskaus.audio_coreaudio import (
    LiveDefaultInput,
    kAudioDevicePropertyDeviceIsAlive,
    kAudioHardwarePropertyDefaultInputDevice,
)
from kuiskaus.audio_device_reset import (
    MAX_TERMINATE_FAILURES,
    MICROPHONE_NEEDS_RESTART,
    MICROPHONE_NOT_SETTLING,
)
from kuiskaus.audio_rebuild_budget import MAX_NEW_EVIDENCE_EXEMPTIONS
from kuiskaus.audio_retry import MICROPHONE_BUSY
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input


def _device_verdict_backoff_open(registry, monkeypatch):
    """Put a clean-teardown DEVICE verdict on the record with the backoff
    window held open, and the streak counting device 88 -- the shape the
    new-evidence exemption answers to (mirrors the ledger-deferral test)."""
    monkeypatch.setattr(reset_module, "_now", lambda: 0.0)
    registry.rebuild.note_rebuild(88)
    registry.last_reset = LiveDefaultInput(88, True)
    registry.needs_restart = True
    registry.terminate_failures = 0
    registry.restart_retry_after = 1e12


def test_a_manual_switch_during_the_backoff_is_picked_up_next_press(
    coreaudio_stub, monkeypatch
):
    """The backoff exists to skip a teardown that is known futile; it must
    not decide which device this session is on. A user who switches to a
    healthy microphone by hand inside the 30 s window is picked up on that
    very press: the guard sits below the live read and the dedupe check, so
    the device is actually read, the stale verdict goes with the evidence,
    and the press is admitted instead of refused.
    """
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_RESET_REBUILDS):
        registry.refresh(_find_device)
    assert registry.needs_restart is True

    # The storm is over and the user picks a different, healthy microphone --
    # WITHOUT the backoff expiring.
    _live_default_input(coreaudio_stub, 250, is_alive=True)
    settled = MagicMock(name="PyAudioInstance-settled")
    module.PyAudio = MagicMock(return_value=settled)

    result = registry.refresh(_find_device)

    assert result.busy is False, "the backoff refused a press that needed no teardown"
    assert result.pyaudio is settled
    assert registry.device_id == 250
    assert registry.press_refusal_reason() != MICROPHONE_NEEDS_RESTART

    # The cross-device count is the bound, so this press -- a rebuild --
    # does not lift a verdict an earlier press earned; the press that does
    # is the one that finds the session already agreeing with the live
    # device, once the window it armed has expired.
    assert registry.needs_restart is True
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    assert registry.refresh(_find_device).busy is False
    assert registry.needs_restart is False
    assert registry.rebuild.count == 0


def test_a_press_that_needs_no_teardown_is_admitted_inside_the_backoff(
    coreaudio_stub, monkeypatch
):
    """The backoff exists to skip one known-futile teardown; it never had a
    right to refuse the press. A press whose live read already matches the
    session is admitted on the spot -- busy=False, no terminate, no
    construction -- so a user still sitting on the microphone that merely
    blinked alive is not told "busy" for the rest of the window."""
    registry, pa, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_RESET_REBUILDS):
        registry.refresh(_find_device)
    assert registry.needs_restart is True
    terminates = pa.terminate.call_count

    # The same device now reads alive: the session already describes it, so
    # there is nothing to reset and nothing to refuse.
    _live_default_input(coreaudio_stub, 77, is_alive=True)
    assert registry.refresh(_find_device).busy is False
    assert pa.terminate.call_count == terminates


def test_the_streak_survives_a_blink_inside_the_window(coreaudio_stub, monkeypatch):
    """The other half of that rule. A device flapping its liveness flag reads
    as "settled" on every other press, so a press inside the window is not
    yet the evidence the rebuild budget was asking for -- handing the budget
    back there would let the flapping device cost a full teardown every other
    press for the life of the process, the churn ``MAX_RESET_REBUILDS``
    exists to bound. The verdict stands until the window expires."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_RESET_REBUILDS * 4):
        registry.refresh(_find_device)
        clock[0] += 1.0  # never reaches the end of the window

    assert registry.needs_restart is True
    assert registry.press_refusal_reason() == MICROPHONE_NOT_SETTLING
    # Bounded: the verdict was earned once and no later press re-earned it.
    assert module.PyAudio.call_count <= reset_module.MAX_RESET_REBUILDS + 1

    # And it is deferred, not abandoned: the window expires, the press runs
    # the teardown it was being refused. That press is itself a rebuild, so
    # the verdict goes with the NEXT press -- the one that finds the session
    # agreeing with the live device (while the window is open, a press is
    # not the settling evidence the budget was asking for).
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    _live_default_input(coreaudio_stub, 88, is_alive=True)
    assert registry.refresh(_find_device).busy is False
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    assert registry.refresh(_find_device).busy is False
    assert registry.needs_restart is False


def test_a_teardown_that_refused_still_says_portaudio_would_not_release(
    coreaudio_stub, monkeypatch
):
    """The counterpart of the honest streak message: this is the verdict that
    HAS earned "PortAudio would not release the device" -- an instance that
    refused ``terminate()`` -- and both user-facing strings must say so."""
    registry, pa, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    _live_default_input(coreaudio_stub, 88)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_TERMINATE_FAILURES):
        registry.refresh(_find_device)

    assert registry.needs_restart is True
    assert registry.terminate_failures > 0
    assert registry.press_refusal_reason() == MICROPHONE_NEEDS_RESTART
    assert registry.unavailable_reason(None) == MICROPHONE_NEEDS_RESTART
    assert MICROPHONE_BUSY not in registry.press_refusal_reason()


def test_a_flapping_device_with_clean_teardowns_does_not_blame_portaudio(
    coreaudio_stub, monkeypatch
):
    """Every teardown in the flapping path terminated cleanly, so the
    session has no evidence PortAudio would not release the device -- and
    the module's own rule is that the claim describes the teardown as it is
    now. The streak verdict says the honest thing (the device is not
    settling, and the backoff retries by itself) instead of sending the user
    to quit the app for a condition that clears on its own.
    """
    registry, pa, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_RESET_REBUILDS * 2):
        registry.refresh(_find_device)

    assert registry.needs_restart is True
    assert registry.terminate_failures == 0  # every teardown released cleanly
    assert pa.terminate.call_count >= reset_module.MAX_RESET_REBUILDS
    assert registry.press_refusal_reason() == MICROPHONE_NOT_SETTLING
    message = registry.unavailable_reason(None)
    assert "settling" in message
    assert "restart" not in message.lower()
    assert MICROPHONE_NEEDS_RESTART not in message


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
    monkeypatch.setattr(reset_module, "_now", lambda: 0.0)
    # A clean-teardown DEVICE verdict on the record -- the shape the
    # exemption answers to -- with the streak counting device 88.
    registry.rebuild.note_rebuild(88)
    registry.last_reset = LiveDefaultInput(88, True)
    registry.needs_restart = True
    registry.terminate_failures = 0
    registry.restart_retry_after = 1e12

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


def _oscillating_liveness(stub, device_id: int) -> None:
    """Report ``device_id`` as the default, with ``isAlive`` alternating on
    every read -- the storm shape that defeats the dedupe guard."""
    counter = {"n": 0}

    def _read(_object_id, address, _qs, _q, _s, _out):
        if address.mSelector == kAudioHardwarePropertyDefaultInputDevice:
            return 0, 4, device_id.to_bytes(4, "little")
        if address.mSelector == kAudioDevicePropertyDeviceIsAlive:
            counter["n"] += 1
            return 0, 4, int(counter["n"] % 2 == 0).to_bytes(4, "little")
        return 1852797079, 0, None  # 'who?' -- unknown property

    stub.AudioObjectGetPropertyData.side_effect = _read


def test_a_persistently_dead_default_rebuilds_once_not_once_per_press(
    coreaudio_stub,
):
    """The stable-pair baseline: the guard engages, and the presses after the
    first cost no teardown and no construction."""
    registry, pa, module = _registry()
    _live_default_input(coreaudio_stub, 77, is_alive=False)
    registry.open(_find_device)

    results = [registry.refresh(_find_device) for _ in range(4)]

    assert all(r.busy is False for r in results)
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # only the open() above


def test_an_engine_verdict_holds_the_window_even_when_the_device_changed(
    coreaudio_stub, monkeypatch
):
    """The exemption the manual switch gets is for the verdict that is ABOUT
    a device. ``rebuild.device`` records the device the rebuild streak counts,
    and a later teardown verdict -- PortAudio refusing an instance -- says
    something about the ENGINE, which no new device excuses. Honouring the
    stale ``rebuild.device`` there would spend the full futile teardown on
    every press, which is the stall the window exists to prevent.
    """
    registry, pa, _module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])

    # One clean rebuild, so the streak now counts device 88.
    _live_default_input(coreaudio_stub, 88)
    registry.refresh(_find_device)
    assert registry.rebuild.device == 88

    # Then the engine turns: every teardown from here on refuses.
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    _live_default_input(coreaudio_stub, 99)
    for _ in range(reset_module.MAX_TERMINATE_FAILURES):
        registry.refresh(_find_device)

    assert registry.needs_restart is True
    assert registry.terminate_failures == reset_module.MAX_TERMINATE_FAILURES
    assert registry.rebuild.device == 88  # stale: the verdict is not about it

    terminates = pa.terminate.call_count
    result = registry.refresh(_find_device)

    assert result.busy is True, "the engine verdict let a press run the futile teardown"
    assert pa.terminate.call_count == terminates
    assert registry.press_refusal_reason() == MICROPHONE_NEEDS_RESTART


def test_an_oscillating_default_is_bounded_rather_than_rebuilding_forever(
    coreaudio_stub,
):
    """With the pair changing on every read the guard cannot
    engage, so consecutive rebuilds have to be counted -- otherwise the
    session churns PortAudio for the life of the process and never says so."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    presses = reset_module.MAX_RESET_REBUILDS * 3

    for _ in range(presses):
        registry.refresh(_find_device)

    assert registry.needs_restart is True
    # Bounded: the rebuilds stop at the budget (plus the one open() above),
    # instead of one per press.
    assert module.PyAudio.call_count <= reset_module.MAX_RESET_REBUILDS + 1


def test_a_settled_device_is_recovered_by_the_retry_after_the_backoff(
    coreaudio_stub, monkeypatch
):
    """The bound defers, it does not give up: once the device stops
    oscillating and the backoff expires, the reset runs again and the
    restart verdict goes away."""
    registry, _pa, module = _registry()
    _live_default_input(coreaudio_stub, 77)
    registry.open(_find_device)
    _oscillating_liveness(coreaudio_stub, 77)
    clock = [0.0]
    monkeypatch.setattr(reset_module, "_now", lambda: clock[0])
    for _ in range(reset_module.MAX_RESET_REBUILDS):
        registry.refresh(_find_device)
    assert registry.needs_restart is True

    _live_default_input(coreaudio_stub, 88, is_alive=True)  # it settled
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    fresh = MagicMock(name="PyAudioInstance-settled")
    module.PyAudio = MagicMock(return_value=fresh)
    result = registry.refresh(_find_device)

    assert result.pyaudio is fresh
    assert registry.device_id == 88
    # The rebuild that recovered the session is still a rebuild, so the
    # verdict it inherits goes when a press finds the session agreeing --
    # not before, which is what keeps a flapping default from earning the
    # budget back on every other press.
    assert registry.needs_restart is True
    clock[0] += reset_module.RESTART_RETRY_SECONDS + 1.0
    assert registry.refresh(_find_device).busy is False
    assert registry.needs_restart is False


def test_the_exhausted_retry_message_bounds_the_foreign_error_text(coreaudio_stub):
    """The menubar's red line is a status item, not a log.

    Every other piece of foreign text this PR puts on screen is bounded
    (``str(arm_error)[:120]`` in the reset's hand-back, the ``[:120]`` on the
    CoreAudio bridge failures) because a native exception can carry an
    arbitrarily long PortAudio/CoreAudio description. This one went through
    unbounded, so a 10 KB PortAudio string landed in the status item verbatim.
    """
    registry, _pa, _module = _registry()
    registry.needs_restart = True
    registry.terminate_failures = MAX_TERMINATE_FAILURES  # the PortAudio verdict

    message = registry.unavailable_reason(OSError("x" * 10_000))

    assert message.startswith(MICROPHONE_NEEDS_RESTART)
    assert len(message) < len(MICROPHONE_NEEDS_RESTART) + 200, (
        f"unbounded native error text in the menubar status item: {len(message)} chars"
    )


def test_the_plain_unavailable_message_bounds_the_foreign_error_text(coreaudio_stub):
    """The other branch that reaches the status item goes through
    ``microphone_unavailable``, which interpolated the error unbounded: the
    ``[:120]`` lived only in the restart-verdict branch above, so the
    generic "Microphone unavailable: ..." line still took a 10 KB PortAudio
    description to the menubar verbatim (issue #72).
    """
    registry, _pa, _module = _registry()
    assert registry.needs_restart is False  # the microphone_unavailable path

    message = registry.unavailable_reason(OSError("x" * 10_000))

    assert message.startswith("Microphone unavailable")
    assert len(message) < 200, (
        f"unbounded native error text in the menubar status item: {len(message)} chars"
    )
