"""The user is told when PortAudio will not come down (issue #72).

A press the device reset cannot complete used to fail in two flavours that
looked identical from outside: "microphone busy" (a thread may still be native
-- defers, and resolves itself) and a poisoned session whose ``terminate()``
never succeeds (never resolves, and nothing ever said so). the harness measured
five consecutive presses returning busy with ``last_error`` None and the
menubar still showing Ready -- the #72 symptom surviving inside the PR written
to remove it.

The budget lives on the registry (``test_audio_device_5.py``); these are the
recorder-side consequences, driven through real presses: the refused press and
the exhausted retry loop each name the restart once the session has given up
on its teardown, and keep their old wording while a retry is still hoped for.
"""

from __future__ import annotations

import threading

from kuiskaus.audio_device_reset import MAX_TERMINATE_FAILURES
from tests.support_audio_recorder import (
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
    _warm_recorder,
)


def _find_device(_pa):
    return 3


def _press_until_the_teardown_gives_up(recorder) -> list[str | None]:
    """Press, re-arm, press again -- and report what each press said.

    The refused warm press hands its (now dead) capture back, so the next
    warm press has to be re-armed the way the worker path's ``recover()``
    does it. Each press runs the real press-side reset, so every refusal of
    the registry's teardown is one the user actually paid for -- and the
    messages come back in the order the user reads them. The re-arm's own
    open runs a reset too, hence the bound rather than an exact count.
    """
    messages: list[str | None] = []
    for _ in range(MAX_TERMINATE_FAILURES * 3):
        assert recorder.start_recording() is False
        messages.append(recorder.last_error)
        if recorder._session.needs_restart:
            return messages
        recorder.set_keep_warm(True)
        assert _wait_until(
            lambda: recorder._warm is not None and recorder._warm.ready
        ), "keep-warm never came back, so the next press is not a warm press"
    raise AssertionError("the session never reached the restart verdict")


def test_a_refused_press_says_busy_while_a_retry_is_still_possible(
    audio_recorder_module, coreaudio_stub
):
    """The control for the next test: one refusal is a deferral, and the
    message must stay "busy" -- a user told to restart the app for a
    transient storm has been told the wrong thing."""
    module = audio_recorder_module
    release = threading.Event()
    recorder, pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    _live_default_input(coreaudio_stub, 77)
    try:
        assert recorder.start_recording() is False
        assert recorder._session.needs_restart is False
        assert "busy" in (recorder.last_error or "").lower()
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_the_press_that_learns_the_engine_will_not_come_down_says_restart(
    audio_recorder_module, coreaudio_stub
):
    """at the press boundary. The press is refused either way,
    so all that changes is whether anything tells the user the microphone is
    not coming back: the presses before the budget say "busy", the one that
    spends it says restart."""
    module = audio_recorder_module
    release = threading.Event()
    recorder, pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    _live_default_input(coreaudio_stub, 77)
    try:
        messages = _press_until_the_teardown_gives_up(recorder)

        assert recorder._session.needs_restart is True
        assert all("busy" in (m or "").lower() for m in messages[:-1]), messages
        last = messages[-1] or ""
        assert "restart" in last.lower(), messages
        assert "busy" not in last.lower(), messages
        # Retried every press, never rebuilt: the budget ends the silence,
        # not the attempts.
        assert pa.terminate.call_count >= MAX_TERMINATE_FAILURES
    finally:
        release.set()
        _cleanup_recorder(recorder)


def test_an_exhausted_retry_on_a_poisoned_session_names_the_restart(
    audio_recorder_module, coreaudio_stub
):
    """The worker path has no press-side refusal to hang the message on: the
    press is admitted and the retry loop is what fails. An exhausted loop on
    a session the registry cannot tear down is the same dead microphone, so
    it must not read as the ordinary "Microphone unavailable" that every
    transient storm also produces."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    recorder = _make_recorder(
        module, init_probe=pa, max_attempts=2, retry_backoff_seconds=(0.01,)
    )
    pa.terminate.side_effect = OSError("Pa_Terminate failed")
    _live_default_input(coreaudio_stub, 77)  # the session never bound to it
    for _ in range(MAX_TERMINATE_FAILURES):
        assert recorder._session.refresh(_find_device).busy is True
    assert recorder._session.needs_restart is True
    pa.open.side_effect = OSError("device gone")
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
        assert "restart" in (recorder.last_error or "").lower()
    finally:
        _cleanup_recorder(recorder)


def test_an_exhausted_retry_says_unavailable_while_a_teardown_is_still_hoped(
    audio_recorder_module,
):
    """Counterpart: while the budget has not been spent, an exhausted retry
    keeps its existing wording -- the restart claim belongs to the state
    where the engine has actually been proven unrecoverable."""
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0)
    pa.open.side_effect = OSError("device gone")
    recorder = _make_recorder(
        module, init_probe=pa, max_attempts=2, retry_backoff_seconds=(0.01,)
    )
    try:
        assert recorder.start_recording() is True
        assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
        assert "Microphone unavailable" in (recorder.last_error or "")
        assert recorder._session.needs_restart is False
    finally:
        _cleanup_recorder(recorder)
