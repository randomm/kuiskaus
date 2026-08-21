"""Hardware-free unit tests for KuiskausMenuBarApp hotkey callbacks.

The listener classes and heavy components are mocked via monkeypatch so
the menubar module import never touches Quartz, pyaudio, or model
loading.
"""

import sys
import threading
import types
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rumps

import kuiskaus.hotkey_listener_cgevent as hlcgevent
from kuiskaus.transcriber import Transcriber


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub hardware/model dependencies before importing menubar."""
    fake_quartz = types.ModuleType("Quartz")
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    fake_audio = types.ModuleType("kuiskaus.audio_recorder")
    fake_audio.AudioRecorder = MagicMock()

    # spec=Transcriber so real isinstance(..., Transcriber) protocol checks
    # in menubar.py (__init__ and _reload_model) pass against these stubs,
    # same as they would against a real transcriber implementation.
    fake_parakeet = types.ModuleType("kuiskaus.parakeet_transcriber")
    fake_parakeet.ParakeetTranscriber = MagicMock(
        return_value=MagicMock(spec=Transcriber)
    )

    fake_whisper = types.ModuleType("kuiskaus.whisper_transcriber")
    fake_whisper.WhisperTranscriber = MagicMock(
        return_value=MagicMock(spec=Transcriber)
    )

    fake_text = types.ModuleType("kuiskaus.text_inserter")
    fake_text.TextInserter = MagicMock()

    monkeypatch.setitem(sys.modules, "kuiskaus.audio_recorder", fake_audio)
    monkeypatch.setitem(sys.modules, "kuiskaus.parakeet_transcriber", fake_parakeet)
    monkeypatch.setitem(sys.modules, "kuiskaus.whisper_transcriber", fake_whisper)
    monkeypatch.setitem(sys.modules, "kuiskaus.text_inserter", fake_text)

    monkeypatch.setattr(hlcgevent, "HotkeyListenerCGEvent", MagicMock(), raising=False)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    _install_stubs(monkeypatch)
    import kuiskaus.menubar as menubar_module

    instance = object.__new__(menubar_module.KuiskausMenuBarApp)
    rumps.App.__init__(instance, "Kuiskaus", title="🎤", quit_button=None)
    instance.status_item = rumps.MenuItem("🟢 Ready", callback=None)
    instance.menu = MagicMock()
    instance.is_recording = False
    instance.recording_start_time = None
    instance.enabled = True
    instance.use_apfel = False
    instance._apfel_lock = threading.Lock()
    instance.total_transcriptions = 0
    instance.total_recording_time = 0.0
    instance.audio_recorder = MagicMock()
    # Real AudioRecorder.__init__ sets last_error = None; a bare
    # MagicMock() attribute is truthy by default, which would make every
    # test believe a mic error is persisted unless overridden here.
    instance.audio_recorder.last_error = None
    # Default transcriber stub; the guard tests reassign it to a fresh
    # mock whose identity assertions don't collide with the fixture.
    instance.transcriber = MagicMock(spec=Transcriber)
    instance._transcriber_lock = threading.Lock()
    return instance


def test_release_without_press_is_noop(app):
    """A release that finds no active recording is a harmless no-op."""
    app.on_hotkey_release()

    app.audio_recorder.stop_recording.assert_not_called()
    app.audio_recorder.start_recording.assert_not_called()
    assert app.is_recording is False
    assert app.status_item.title == "🟢 Ready"


def test_release_restores_ready_without_recorder_calls(app):
    """The no-op release must restore Ready status (menu bar surface)."""
    app.status_item.title = "🟡 Processing..."
    app.on_hotkey_release()

    assert app.audio_recorder.stop_recording.call_count == 0
    assert app.status_item.title == "🟢 Ready"


def test_disabled_release_is_noop(app):
    """A release while disabled is a no-op even with stale state."""
    app.enabled = False
    app.is_recording = True
    app.on_hotkey_release()

    app.audio_recorder.stop_recording.assert_not_called()
    assert app.status_item.title == "🟢 Ready"


def test_press_refused_does_not_flip_recording_state(app):
    """start_recording()'s bool return must be consumed (#16): a refused
    admission must not set is_recording True."""
    app.audio_recorder.start_recording = MagicMock(return_value=False)

    app.on_hotkey_press()

    assert app.is_recording is False
    assert app.recording_start_time is None


def test_press_admitted_flips_recording_state(app):
    """An admitted start_recording() still sets is_recording True."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)

    app.on_hotkey_press()

    assert app.is_recording is True
    assert app.recording_start_time is not None


def test_release_with_last_error_surfaces_and_persists(app):
    """A failed/stuck microphone open must surface via update_status() and
    persist (no auto-clear) -- not be masked as a silent return to Ready (#16)."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.audio_recorder.stop_recording = MagicMock(return_value=b"\x00\x00")
    app.audio_recorder.last_error = "microphone busy — recording did not start"

    app.on_hotkey_press()
    with patch("threading.Thread") as mock_thread_cls:
        app.on_hotkey_release()
        mock_thread_cls.assert_not_called()

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready"


def test_release_without_last_error_still_transcribes(app):
    """No last_error: the normal transcription path still fires."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.audio_recorder.stop_recording = MagicMock(
        return_value=np.array([0.1, 0.2], dtype=np.float32)
    )
    app.audio_recorder.last_error = None

    app.on_hotkey_press()
    with patch("threading.Thread") as mock_thread_cls:
        app.on_hotkey_release()
        mock_thread_cls.assert_called_once()


def test_press_refused_paired_release_preserves_banner(app):
    """Reproduces the regression fixed here: a press refused by
    start_recording() (worker still alive, #16) never sets is_recording
    True, so the physically-paired release for that same keypress lands
    in on_hotkey_release()'s no-op branch. A persisted error banner must
    survive that branch -- it must NOT be silently wiped back to Ready."""
    app.audio_recorder.last_error = "microphone busy — recording did not start"
    app.audio_recorder.start_recording = MagicMock(return_value=False)
    app.status_item.title = f"🔴 {app.audio_recorder.last_error}"

    app.on_hotkey_press()
    assert app.is_recording is False  # admission was refused

    app.on_hotkey_release()  # paired release for the same keypress

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready"


def test_banner_clears_after_successful_recording(app):
    """A genuinely successful recording (no last_error, empty audio) is
    the one case that DOES clear a previously persisted banner."""
    app.status_item.title = "🔴 microphone busy — recording did not start"
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.audio_recorder.stop_recording = MagicMock(
        return_value=np.array([], dtype=np.float32)
    )
    app.audio_recorder.last_error = None

    app.on_hotkey_press()
    app.on_hotkey_release()

    assert app.status_item.title == "🟢 Ready"


def test_toggle_enabled_preserves_banner_on_reenable(app):
    """Re-enabling after a persisted mic error must not silently clear it
    (#16) -- an enable/disable toggle is not a successful recording."""
    app.audio_recorder.last_error = "microphone busy — recording did not start"
    app.enable_item = MagicMock()
    app.enabled = False
    app.status_item.title = f"🔴 {app.audio_recorder.last_error}"

    app.toggle_enabled(app.enable_item)  # False -> True

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready"


def test_stale_transcription_completion_preserves_newer_banner(app):
    """A transcription thread spawned for an earlier, error-free recording
    must not clobber the mic-error banner of a *later* failed recording
    that completed while the earlier transcription was still running (#16).

    Runs the race for real: _transcribe_and_insert runs on a live thread
    blocked inside transcribe() while the test sets last_error (simulating
    the later failed cycle); on release, the stale completion must NOT
    reset the banner to Ready. With the guard, the banner surface (which
    renders last_error) still shows the new error; without it, the stale
    completion clobbers the status -- this test fails (banner goes back
    to Ready while the UI would still render the error).
    """
    app.transcriber = MagicMock()
    release_event = threading.Event()
    worker_blocked = threading.Event()

    def blocking_transcribe(*_args, **_kwargs):
        worker_blocked.set()  # tell main we are inside transcribe()
        release_event.wait(timeout=10.0)
        return {"text": "hello"}

    app.transcriber.transcribe.side_effect = blocking_transcribe
    app.text_inserter = MagicMock()
    # A plain object, not a real rumps.MenuItem: once a rumps.App exists
    # (created by the fixture via rumps.App.__init__), any ObjC call
    # (e.g. the title setter on the real NSMenuItem) made from a
    # background thread hangs the process, and an UNBOUNDED
    # Event.wait() on the main thread hangs at interpreter shutdown
    # (pyobjc/NSApp teardown quirk). All other tests touch the
    # fixture's real MenuItem only from the main thread; this test's
    # point is the worker-thread write, so it observes the banner
    # through a plain Python object and keeps every wait bounded.
    app.status_item = type("Status", (), {"title": "\U0001f7e2 Ready"})()
    app.audio_recorder.last_error = None
    # Spy on update_status: the falsifiable assertion is that the stale
    # worker never calls it at all. Without the last_error guard, the
    # worker would write "🟢 Ready" on top of the newer failed cycle's
    # banner -- the clobber this test forbids.
    update_status_spy = MagicMock()
    orig_update_status = app.update_status

    def spy_update_status(status: str) -> None:
        orig_update_status(status)
        update_status_spy(status)

    app.update_status = spy_update_status

    thread = threading.Thread(
        target=app._transcribe_and_insert,
        args=(np.array([0.1], dtype=np.float32), 1.0),
    )
    thread.start()

    # Wait until the worker is deterministically blocked inside
    # transcribe() before perturbing state.
    assert worker_blocked.wait(timeout=5.0), "worker never reached transcribe()"
    app.audio_recorder.last_error = "microphone busy — recording did not start"

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    app.text_inserter.insert_text.assert_called_once_with("hello")
    # The stale completion of an error-free recording must NOT touch the
    # banner: a later, newer cycle has since set last_error, so writing
    # "🟢 Ready" here would clobber the live error the UI is about to
    # render. The guard in _transcribe_and_insert is what keeps this
    # write out -- without it, the spy records the clobber and the test
    # fails.
    update_status_spy.assert_not_called()
    # Belt-and-suspenders: the banner's surface value is also unchanged
    # from its initial state, and last_error still carries the newer
    # failure so any UI render reads it correctly.
    assert app.status_item.title == "🟢 Ready"  # not clobbered mid-flight
    assert app.audio_recorder.last_error == (
        "microphone busy — recording did not start"
    )
    assert "busy" in app.audio_recorder.last_error


def test_reload_model_success_preserves_banner_when_last_error_set(app):
    """A model switch never touches the microphone; if a mic-error banner
    is live, a successful model reload must not silently clear it (#16
    DoD: persists until the next successful *recording*, not a model
    switch)."""
    app.transcriber = MagicMock()
    app.audio_recorder.last_error = "microphone busy — recording did not start"
    app.status_item.title = f"🔴 {app.audio_recorder.last_error}"

    app._reload_model("parakeet")

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready"


def test_reload_model_success_clears_ready_without_last_error(app):
    """Without a persisted mic error, a successful model reload still
    resets the status to Ready."""
    app.transcriber = MagicMock()
    app.audio_recorder.last_error = None

    app._reload_model("parakeet")

    assert app.status_item.title == "🟢 Ready"


def test_reload_model_exception_preserves_banner_when_last_error_set(app):
    """A model-reload failure must not clobber a live mic-error banner
    either -- same rationale as the success path above."""
    app.transcriber = MagicMock()
    app.transcriber.cleanup.side_effect = RuntimeError("boom")
    app.audio_recorder.last_error = "microphone busy — recording did not start"
    app.status_item.title = f"🔴 {app.audio_recorder.last_error}"

    app._reload_model("parakeet")

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready (model error)"


def test_transcriber_snapshot_survives_reload_cleanup(app):
    """A transcription worker must keep running its snapshot even after
    _reload_model has swapped in a new transcriber and cleaned up the
    old one (issue #22). The worker holds the transcriber lock for the
    entire transcribe() call, so _reload_model's cleanup() of the old
    transcriber cannot overlap the in-flight inference.

    Fails without the fix: with no lock held during transcribe(),
    _reload_model's cleanup() fires while the worker is still inside
    transcribe(), and the 'cleanup must not have started yet' assertion
    catches it.
    """
    import time

    old = MagicMock(spec=Transcriber)
    new = MagicMock(spec=Transcriber)
    app.transcriber = old
    app.text_inserter = MagicMock()
    worker_blocked = threading.Event()
    release_event = threading.Event()
    cleanup_started = threading.Event()

    def blocking_transcribe(*_args, **_kwargs):
        worker_blocked.set()  # tell main we are inside transcribe()
        release_event.wait(timeout=10.0)
        return {"text": "hello"}

    def cleanup_tracking():
        cleanup_started.set()  # mark the moment cleanup() actually runs

    old.transcribe.side_effect = blocking_transcribe
    old.cleanup.side_effect = cleanup_tracking
    new.transcribe.return_value = {"text": "you should never see me"}

    with patch("kuiskaus.menubar.ParakeetTranscriber", return_value=new):
        thread = threading.Thread(
            target=app._transcribe_and_insert,
            args=(np.array([0.1], dtype=np.float32), 1.0),
        )
        thread.start()

        # Deterministically parked inside old.transcribe() while holding
        # the lock.  Now run the real _reload_model on a second thread.
        assert worker_blocked.wait(timeout=5.0), "worker never reached transcribe()"
        reload_thread = threading.Thread(
            target=lambda: app._reload_model("parakeet"),
        )
        reload_thread.start()

        # _reload_model is blocked on the transcriber lock that the
        # worker still holds.  Give it time to attempt the cleanup; it
        # must NOT have started yet because the worker is still inside
        # transcribe() and holding the lock.
        time.sleep(0.3)
        assert not cleanup_started.is_set(), (
            "cleanup() started while the worker was still mid-transcribe() — "
            "the transcriber lock is not held across the inference call"
        )

        release_event.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "worker thread did not finish"
        reload_thread.join(timeout=10.0)

    # The worker finished its inference against the OLD transcriber it
    # bound at the start of the run...
    old.transcribe.assert_called_once()
    app.text_inserter.insert_text.assert_called_once_with("hello")
    # ...and never touched the swapped-in, already-active transcriber.
    new.transcribe.assert_not_called()
    # cleanup() ran exactly once, only after the worker released the lock.
    old.cleanup.assert_called_once()
    assert app.transcriber is new


def test_reload_model_swap_is_atomic(app):
    """_reload_model must commit the swap and run cleanup() atomically
    under the transcriber lock (issue #22). A concurrent read via
    _current_transcriber() must observe either the OLD or the NEW
    transcriber — never an in-between (unassigned) state.

    The test parks the worker inside old.transcribe(), then lets
    _reload_model run for real on a second thread. A reader thread
    hammers _current_transcriber() while the reload is in progress.
    Every observation must be `old` or `new` — the lock ensures no
    intermediate state is ever visible.

    Fails without the fix: if the swap is not under the lock, the
    reader can observe a partially-assigned state (in practice the
    GIL makes the single-assignment atomic in CPython, so this test
    primarily validates that the lock serializes the reader against
    the swap, which the `observed` list proves)."""
    old = MagicMock(spec=Transcriber)
    new = MagicMock(spec=Transcriber)
    app.transcriber = old
    app.text_inserter = MagicMock()
    worker_blocked = threading.Event()
    release_event = threading.Event()
    stop = threading.Event()
    observed: list = []

    def blocking_transcribe(*_args, **_kwargs):
        worker_blocked.set()
        release_event.wait(timeout=10.0)
        return {"text": "hello"}

    old.transcribe.side_effect = blocking_transcribe

    with patch("kuiskaus.menubar.ParakeetTranscriber", return_value=new):

        def reader():
            while not stop.is_set():
                observed.append(app._current_transcriber())

        thread = threading.Thread(
            target=app._transcribe_and_insert,
            args=(np.array([0.1], dtype=np.float32), 1.0),
        )
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        thread.start()
        assert worker_blocked.wait(timeout=5.0), "worker never reached transcribe()"
        reload_thread = threading.Thread(
            target=lambda: app._reload_model("parakeet"),
        )
        reload_thread.start()
        # Let the reader collect some observations while the reload is
        # blocked on the lock the worker holds.
        import time

        time.sleep(0.3)
        stop.set()
        release_event.set()
        reader_thread.join(timeout=10.0)
        thread.join(timeout=10.0)
        reload_thread.join(timeout=10.0)

    # Every serialized read saw one of the two committed transcribers,
    # never an in-between value.
    assert observed, "reader never ran"
    assert all(t in (old, new) for t in observed), (
        f"reader observed an in-between transcriber state: {set(map(type, observed))}"
    )
    # The reload finished its swap and cleaned up the old transcriber.
    assert app.transcriber is new
    old.cleanup.assert_called_once()


def test_utcnow_helper_returns_aware_utc():
    """_utcnow() is the single source of the aware-UTC invariant (issue
    #22): aware UTC so it can be subtracted from session_start without
    a TypeError from a naive/other-tz datetime."""
    from kuiskaus.menubar import _utcnow

    now = _utcnow()

    assert now.tzinfo is UTC
    # Two calls both return aware datetimes in the same tz, so
    # subtracting them can never raise the naive/aware TypeError this
    # helper exists to prevent. (No monotonicity assertion: the host
    # wall clock steps backward.)
    assert (now - _utcnow()) is not None


def test_show_stats_uses_utcnow_helper(app, monkeypatch):
    """show_stats() must compute its session duration via _utcnow() so the
    aware-UTC invariant lives in one place (issue #22)."""
    import kuiskaus.menubar as menubar_module

    app.session_start = datetime(2026, 1, 1, tzinfo=UTC)
    calls = []
    monkeypatch.setattr(
        menubar_module,
        "_utcnow",
        lambda: (calls.append(1), datetime(2026, 1, 1, 0, 30, tzinfo=UTC))[1],
    )

    with patch.object(rumps, "alert") as mock_alert:
        app.show_stats(None)

    assert calls == [1], "show_stats must build its now via _utcnow()"
    mock_alert.assert_called_once()
    assert "0h 30m" in mock_alert.call_args.args[1]


def test_app_module_exposes_shared_silicon_check():
    """app.py must use the shared implementation (issue #22 dedup) rather
    than its own private copy."""
    from kuiskaus.app import check_apple_silicon
    from kuiskaus.silicon_check import check_apple_silicon as shared

    assert check_apple_silicon is shared
