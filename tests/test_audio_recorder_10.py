"""The worker's own head must not be able to strand the recorder.

``_recording_worker`` opens by joining its orphaned predecessor, running the
live-default reset and re-arming keep-warm -- calls into code that reaches
CoreAudio and the thread ledgers. None of that was guarded, so a raise there
left ``recording`` True with no ``last_error`` and no live worker: later
presses were refused as stale and the user saw a recorder that had stopped
answering. Everything else in the worker already has a generation-gated
failure path; the head had none.

The head's reset also has to *reach the session*. The worker holds no registry
of its own, and resolving one from the instance it was handed means the reset
silently does nothing in the one state that most needs it -- ``pyaudio`` is
None because a half-finished teardown unbound it, with a live instance still
registered -- which is how the fix reproduced issue #72's own symptom.
The last test here drives that state.
"""

from kuiskaus.audio_device_reset import MAX_TERMINATE_FAILURES
from tests.support_audio_recorder import (
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _wait_until,
)


def _raising_reset(module, monkeypatch, error: BaseException):
    """Make the worker's session reset raise, and hand back nothing else."""

    def explode(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(module, "refresh_recorder_session", explode)


def test_a_raising_session_reset_fails_the_press_instead_of_stranding_it(
    audio_recorder_module, monkeypatch
):
    """The reset is native code behind a PyObjC bridge; it may raise. The
    press has to come back failed, with the error surfaced -- not stay
    "recording" over a worker that already died."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(module, init_probe=pa)
    _raising_reset(module, monkeypatch, OSError("CoreAudio exploded"))
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0), (
            "recording stayed True after the worker's head raised"
        )
        assert recorder.last_error is not None, "no error was ever surfaced"
        assert recorder.recording_thread is None
    finally:
        _cleanup_recorder(recorder)


def test_a_raising_keep_warm_recovery_fails_the_press_too(
    audio_recorder_module, monkeypatch
):
    """Same boundary, second call: keep-warm recovery runs right after the
    reset and reaches the same native paths."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(module, init_probe=pa)

    def explode(*_args, **_kwargs):
        raise RuntimeError("ledger wedged")

    monkeypatch.setattr(recorder._session.warm, "recover", explode)
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
        assert recorder.last_error is not None
    finally:
        _cleanup_recorder(recorder)


def test_the_failure_path_is_generation_gated_like_every_other_worker_write(
    audio_recorder_module, monkeypatch
):
    """A late failure from a superseded worker must not wipe the current
    take's state -- the guard shares the ``_generation == my_gen`` gate the
    rest of the worker writes under."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(module, init_probe=pa)
    _raising_reset(module, monkeypatch, OSError("CoreAudio exploded"))
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
        stale_gen = recorder.current_generation

        # A newer generation takes over, and its own worker dies the same
        # way. Both presses fail on their own merits; neither writes over
        # the other's state.
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
        assert recorder.current_generation > stale_gen
        assert recorder.last_error is not None
    finally:
        _cleanup_recorder(recorder)


def _find_device(_pa: object) -> int:
    return 3


def test_a_worker_press_finishes_a_teardown_the_reset_left_half_done(
    audio_recorder_module, coreaudio_stub
):
    """The state ``terminate_all()`` leaves when one instance refuses: the
    session names no instance (it just terminated the live one) while a live
    one is still registered holding PortAudio's init count above zero. Only a
    later press's reset can finish the job -- and a reset that resolves the
    session from the instance it was handed is handed ``None`` right here, so
    it returns without touching anything.

    Measured over repeated presses in that state with the indirection in
    place: 0 further ``terminate()`` attempts, ``terminate_failures`` frozen
    at 1, ``needs_restart`` never reached, one fresh ``PyAudio()`` per press
    each replaying PortAudio's frozen device list, and ``last_error`` stuck on
    the plain "Microphone unavailable" for the life of the process. The opens
    fail *because* the snapshot is frozen, so nothing re-binds the session and
    the state does not heal from inside; the same registry state reached
    through the keep-warm path recovers, because that route asks the recorder
    for the session rather than the instance. The asymmetry is the tell.
    """
    module = audio_recorder_module
    live = _make_pyaudio_instance(0, rate=16000.0)
    poison = _make_pyaudio_instance(7, rate=16000.0)
    poison.terminate.side_effect = OSError("Pa_Terminate failed")
    # Every instance built from here on fails to open -- PortAudio still
    # lists the device the half-teardown was meant to leave behind.
    frozen = _make_pyaudio_instance(5, rate=16000.0)
    frozen.open.side_effect = OSError("device gone: stale snapshot")

    _live_default_input(coreaudio_stub, 77)  # the session binds to device 77
    recorder = _make_recorder(
        module,
        init_probe=live,
        extra_pa=frozen,
        max_attempts=2,
        retry_backoff_seconds=(0.01,),
    )
    recorder._session.register(poison)  # a second instance this session built

    # The half-teardown, through the real reset: the default moved, the live
    # instance terminated, the poison one refused.
    _live_default_input(coreaudio_stub, 88)
    assert recorder._session.refresh(_find_device).busy is True
    assert recorder.pyaudio is None
    assert poison in recorder._session._instances
    attempts = poison.terminate.call_count

    try:
        for _ in range(MAX_TERMINATE_FAILURES + 1):
            assert recorder.start_recording() is True
            assert _wait_until(lambda: recorder.recording is False, timeout=8.0)

        # The worker's head reached the registry: the instance that would not
        # die was asked again, the refusals added up to a verdict, and that
        # verdict is what the user reads instead of a bare "unavailable".
        assert poison.terminate.call_count > attempts, (
            "the worker press never ran the session's reset: the poison "
            "instance was never retried and PortAudio stayed initialised "
            "on its frozen device list"
        )
        assert recorder._session.needs_restart is True
        assert recorder.last_error is not None
        assert "restart" in recorder.last_error.lower()
    finally:
        _cleanup_recorder(recorder)


def test_the_worker_head_guard_logs_what_failed(
    audio_recorder_module, monkeypatch, capsys
):
    """A silent failure path is how this class of bug stays invisible: the
    raise has to leave a line in the log the user can paste."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(module, init_probe=pa)
    _raising_reset(module, monkeypatch, OSError("CoreAudio exploded"))
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
        assert "CoreAudio exploded" in capsys.readouterr().out
    finally:
        _cleanup_recorder(recorder)
