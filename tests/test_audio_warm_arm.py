"""The #72 adversarial findings, as failing-then-passing pairs.

Round 1:

1. Orphaned warm capture: ``set_keep_warm`` armed with a lock-free
   check-then-act on ``recorder._warm``. #72 gave arming two extra
   caller threads (the worker's ``recover`` re-arm, a hotkey press's
   reset re-arm), so two threads could both pass the None check and
   both start a capture; the loser is referenced by nobody and absent
   from ``WarmKeep._blockers``, so ``terminate_all()`` drives PortAudio
   out from under it. Exactly one started capture must survive any
   interleaving.

2. Stale worker name in the reset gate: ``_open_warm_stream`` hands
   ``_refresh_pyaudio_session`` the worker it captured BEFORE a join of
   up to 30 s. If that worker finished and a new one became current in
   the meantime, the "never reset under a live worker" gate trusted the
   dead name and terminated the session under the live worker. The gate
   must consult the recorder's CURRENT worker at gate time.

Round 2:

3. The reset's re-arm overrode the menu. It went through
   ``set_keep_warm(True)``, which WRITES the keep-warm request as well as
   satisfying it, and ``reset_session`` runs outside the recorder lock for a
   whole terminate-all + ``Pa_Initialize`` + re-enumeration -- long enough
   for the user to uncheck the item. The uncheck was erased, a capture was
   re-armed, and the microphone was left open behind a menu item showing
   OFF. ``WarmArm.rearm`` satisfies the request the user has; only the menu
   is allowed to make one.

4. A capture whose thread could not be created stayed installed, and (with
   the round-2 gate fix, which treats an installed capture with no thread as
   arming) would then refuse every later ``terminate_all()`` for the life of
   the process.
"""

import threading
from unittest.mock import MagicMock

import pytest

from kuiskaus.audio_warm_arm import WarmArm
from kuiskaus.audio_warm_keep import WarmKeep
from tests.support_audio_recorder import (
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
)


def test_concurrent_arming_from_two_threads_starts_exactly_one_capture(
    audio_recorder_module,
):
    """Finding 1: menu thread ``set_keep_warm(True)`` racing a worker
    thread ``_rearm_warm(True)`` from cold must leave exactly one
    started, not-stopped capture -- the installed one. The loser of a
    serialised arm is either never started or cleanly dropped."""
    module = audio_recorder_module
    recorder = _make_recorder(module)

    constructed: list = []
    second_arrived = threading.Event()
    release = threading.Event()

    class _RacingCapture:
        """Stand-in for WarmCapture that makes the unserialised window
        deterministic: it parks inside the constructor (before the
        check-then-act can assign) until released, so a lock-free second
        armer provably sees ``_warm`` still None and constructs too.
        Every start/stop is recorded instead of spawning a thread."""

        def __init__(self, _open_stream, _chunk_size):
            constructed.append(self)
            if len(constructed) >= 2:
                second_arrived.set()
            release.wait(timeout=1.5)
            self.started = False
            self.stopped = False
            self.capture_rate = None

        def start(self):
            self.started = True

        def stop(self, timeout: float = 2.0) -> bool:
            self.stopped = True
            return True

    module.WarmCapture = _RacingCapture
    errors: list = []

    def _run(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - thread boundary, surfaced via assertion
            errors.append(exc)

    # Thread A (the menu) enters the arm first and is held in the
    # constructor; Thread B (a #72 re-arm caller) then runs its whole
    # _rearm_warm against the still-unassigned slot.
    t_menu = threading.Thread(
        target=_run, args=(lambda: recorder.set_keep_warm(True),), daemon=True
    )
    t_menu.start()
    assert _wait_until(lambda: len(constructed) >= 1, timeout=5.0), "no capture armed"
    t_rearm = threading.Thread(
        target=_run, args=(lambda: recorder._rearm_warm(True),), daemon=True
    )
    t_rearm.start()
    # Unserialised arming: B passes the None check while A is still in
    # its constructor and arrives too. Serialised arming: B queues
    # behind the lock and never constructs a second capture.
    second_arrived.wait(timeout=1.0)
    release.set()
    t_menu.join(timeout=10.0)
    t_rearm.join(timeout=10.0)
    assert not errors, errors

    live_started = [c for c in constructed if c.started and not c.stopped]
    assert len(live_started) == 1, (
        f"{len(live_started)} started-and-live captures from "
        f"{len(constructed)} constructions: an orphaned warm thread is "
        "invisible to WarmKeep, so terminate_all() may run under it"
    )
    assert live_started[0] is recorder._warm


def _arm_fixture():
    """A WarmArm over a fresh WarmKeep, plus a capture factory recording
    every capture it built."""
    keep = WarmKeep()
    arm = WarmArm(keep, threading.Lock())
    built: list[MagicMock] = []

    def make() -> MagicMock:
        capture = MagicMock(name="warm-capture")
        capture.capture_rate = 16000
        built.append(capture)
        return capture

    return keep, arm, built, make


def test_rearm_never_states_a_keep_warm_request_the_user_never_made():
    """Round 2 finding 1 (unit): the re-arm hand-back must satisfy the
    request, not make one. Only ``set_enabled`` -- the menu's path -- writes
    ``WarmKeep.enabled``."""
    keep, arm, built, make = _arm_fixture()

    arm.rearm(make)  # nothing was ever requested -> nothing is armed
    assert built == []
    assert arm.capture is None
    assert keep.wanted() is False

    keep.want(True)  # the user asks for keep-warm
    arm.rearm(make)
    assert built == [arm.capture]

    arm.set_enabled(make, lambda _rate: None, False)  # the user unchecks it
    built.clear()
    arm.rearm(make)  # a reset's late hand-back lands after the uncheck
    assert built == [], "a late re-arm resurrected keep-warm behind the menu"
    assert arm.capture is None
    assert keep.wanted() is False


def test_rearm_on_an_installed_capture_spends_the_pending_recovery():
    """Round 2 finding 1, the counterpart of ``want(True)``'s flag clearing:
    ``rearm`` no longer routes through ``want``, so it must consume the
    recovery itself or the next worker press re-arms a SECOND capture over a
    working one."""
    keep, arm, built, make = _arm_fixture()
    keep.want(True)
    arm.rearm(make)
    assert len(built) == 1

    keep.need_recovery()
    arm.rearm(make)  # already armed for the request -> no second capture

    assert len(built) == 1
    assert keep._recover_pending is False


def test_a_capture_whose_thread_could_not_start_is_not_left_installed():
    """Round 2 finding 2's companion: the gate now treats an installed
    capture with no thread as arming, so a ``start()`` that failed (thread
    creation does fail under a thread-limit storm) must undo the install --
    otherwise a capture that can never read would refuse every later
    ``terminate_all()`` for the life of the process."""
    keep, arm, _built, _make = _arm_fixture()
    broken = MagicMock(name="capture")
    broken.capture_rate = 16000
    broken.start.side_effect = RuntimeError("can't start new thread")

    with pytest.raises(RuntimeError):
        arm.set_enabled(lambda: broken, lambda _rate: None, True)

    assert arm.capture is None
    assert keep.installed_may_be_native() is False
    assert keep.clear() is True
    assert keep.drop(broken) is True  # nothing left holding the engine open


def _keep_warm_stays_off(recorder) -> bool:
    """No capture appeared in the moments after the press (a late recovery
    or a re-arm landing out of order would arm one)."""
    for _ in range(20):
        threading.Event().wait(timeout=0.01)
        if recorder._warm is not None:
            return False
    return True


def test_a_reset_does_not_resurrect_keep_warm_the_user_unchecked_during_it(
    audio_recorder_module, coreaudio_stub
):
    """Round 2 finding 1, end to end through the public API.

    The uncheck is fired from inside the reset's own ``PyAudio()``
    construction -- a moment genuinely inside the window a menu click can
    land in (the reset runs outside the recorder lock, and this is the
    rebuild at its tail). Before the fix the capture came back and the
    session's mirror of the menu said ON, so the mic indicator stayed lit
    over an unchecked "Keep mic warm".
    """
    module = audio_recorder_module
    release = threading.Event()
    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.return_value = _streaming_stream(release)
    rebuild = _make_pyaudio_instance(5, rate=16000.0)
    rebuild.open.return_value = _streaming_stream(release)
    recorder = _make_recorder(module, rebuild, init_probe=pa, extra_pa=rebuild)
    recorder.set_keep_warm(True)
    assert _wait_until(lambda: recorder._warm.ready)
    warm = recorder._warm
    unchecked: list[bool] = []

    construct = module.pyaudio.PyAudio

    def _construct_while_the_menu_clicks(*args, **kwargs):
        if not unchecked:
            unchecked.append(True)
            recorder.set_keep_warm(False)  # the user gives up mid-reset
        return construct(*args, **kwargs)

    module.pyaudio.PyAudio = _construct_while_the_menu_clicks
    _live_default_input(coreaudio_stub, 77)  # the default moved
    try:
        assert recorder.start_recording() is True
        assert recorder._warm is None, (
            "the reset re-armed a keep-warm capture the user had just "
            "unchecked, leaving the mic open behind an unchecked menu item"
        )
        worker = recorder.recording_thread
        assert worker is not None  # no warm capture left -> the worker path
        assert _wait_until(lambda: recorder.pyaudio is rebuild, timeout=8.0)

        assert unchecked == [True]  # the reset really did run
        pa.terminate.assert_called_once()  # ... and it rebuilt
        assert warm._stop.is_set()  # the capture it displaced was stopped
        assert recorder._session.warm.wanted() is False
        assert _keep_warm_stays_off(recorder)
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_reset_gate_checks_the_current_worker_not_the_joined_name(
    audio_recorder_module, coreaudio_stub
):
    """Finding 2: a FINISHED W1 handed in as the joined worker while a
    LIVE W2 is the recorder's current worker must read busy -- no
    terminate, no rebuild. The gate re-reads the recorder's current
    worker at gate time, not the name captured before the join."""
    module = audio_recorder_module
    recorder = _make_recorder(module)
    pa = recorder.pyaudio
    assert pa is not None

    # W1: the worker whose name was captured before the join -- done.
    w1 = threading.Thread(target=lambda: None, name="W1")
    w1.start()
    w1.join(timeout=5.0)
    assert not w1.is_alive()

    # W2: the current worker, parked in a native call.
    hold = threading.Event()
    w2 = threading.Thread(target=hold.wait, name="W2", daemon=True)
    w2.start()
    try:
        recorder.recording_thread = w2
        _live_default_input(coreaudio_stub, 77)  # the default moved

        recorder._refresh_pyaudio_session(w1)

        pa.terminate.assert_not_called()
        assert recorder.pyaudio is pa
        assert module.pyaudio.PyAudio.call_count == 1  # nothing rebuilt
    finally:
        hold.set()
        w2.join(timeout=5.0)

    # The busy verdict is a deferral, not a wedge: with both named
    # workers gone, the next reset proceeds and terminates.
    recorder._refresh_pyaudio_session(w1)
    pa.terminate.assert_called_once()
    assert module.pyaudio.PyAudio.call_count == 2
