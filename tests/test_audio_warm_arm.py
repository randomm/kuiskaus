"""The two #72 adversarial findings, as failing-then-passing pairs.

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
"""

import threading

from tests.support_audio_recorder import (
    _live_default_input,
    _make_recorder,
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
