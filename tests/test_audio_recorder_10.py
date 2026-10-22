"""The worker's own head must not be able to strand the recorder.

``_recording_worker`` opens by joining its orphaned predecessor, running the
live-default reset and re-arming keep-warm -- calls into code that reaches
CoreAudio and the thread ledgers. None of that was guarded, so a raise there
left ``recording`` True with no ``last_error`` and no live worker: later
presses were refused as stale and the user saw a recorder that had stopped
answering. Everything else in the worker already has a generation-gated
failure path; the head had none.
"""

from tests.support_audio_recorder import (
    _cleanup_recorder,
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
