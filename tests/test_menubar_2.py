"""test_menubar tests, part 2 of 2 (issue #70)."""

import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rumps

from kuiskaus.transcriber import Transcriber
from tests.support_menubar import (
    _install_stubs,
    _parked_worker,
)


def test_transcriber_snapshot_survives_reload_cleanup(app):
    """A transcription worker must keep running its snapshot even after
    _reload_model has swapped in a new transcriber and cleaned up the
    old one (issue #22). The worker holds the transcriber lock for the
    entire transcribe() call, so _reload_model's cleanup() of the old
    transcriber cannot overlap the in-flight inference.

    Deterministic by construction: the reload thread is started while
    the worker is parked inside transcribe(), and the mock cleanup()
    records — at the exact moment it runs — whether the worker had
    already returned from transcribe() (worker_done). In the fixed
    source the reload blocks on the transcriber lock the worker holds
    across transcribe(), so cleanup() can only run after the worker
    released it, i.e. after worker_done is set: the recorded value is
    always True and the assertion passes. If the lock-across-
    transcribe guard is removed, the reload needs no lock to swap and
    clean up, so cleanup() can run while the worker is still inside
    transcribe(): the recorded value is False and the assertion fails.
    The recording happens inside cleanup(), so there is no check window
    for the main thread to race in — the ordering is captured at the
    point where it happens.

    Fails without the fix: with no lock held during transcribe(),
    _reload_model's cleanup() fires while the worker is still inside
    transcribe(), and the 'cleanup must have seen worker_done' assertion
    catches it.
    """
    old = MagicMock(spec=Transcriber)
    new = MagicMock(spec=Transcriber)
    app.transcriber = old
    scaffold = _parked_worker(app, old)
    cleanup_gate = threading.Event()
    # Recorded by the mock cleanup() at the moment it runs: had the
    # worker already returned from transcribe()?
    cleanup_saw_worker_done = []

    def gated_cleanup():
        cleanup_saw_worker_done.append(scaffold["worker_done"].is_set())
        cleanup_gate.wait(timeout=10.0)

    old.cleanup.side_effect = gated_cleanup
    new.transcribe.return_value = {"text": "you should never see me"}

    with patch("kuiskaus.model_reload.ParakeetTranscriber", return_value=new):
        scaffold["worker"].start()
        # Deterministically parked inside old.transcribe() while holding
        # the transcriber lock (fixed source), or just parked inside
        # old.transcribe() without the lock (broken source).
        assert scaffold["parked"].wait(timeout=5.0), "worker never reached transcribe()"
        # Start the reload while the worker is still inside transcribe():
        # in the fixed source it blocks on the transcriber lock; in the
        # broken source it proceeds straight to the swap and cleanup().
        reload_thread = threading.Thread(
            target=lambda: app._reload_model("parakeet"),
        )
        reload_thread.start()
        # The worker now finishes transcribe() and releases the lock;
        # the reload (blocked on the lock the whole time) commits its
        # swap and then runs cleanup() — which records worker_done and
        # blocks on the gate.
        scaffold["release_event"].set()
        assert scaffold["worker_done"].wait(timeout=10.0), "worker never finished"
        # Let the reload finish its (now unblocked) cleanup.
        cleanup_gate.set()
        reload_thread.join(timeout=10.0)
        scaffold["worker"].join(timeout=10.0)

    # The worker finished its inference against the OLD transcriber it
    # bound at the start of the run...
    old.transcribe.assert_called_once()
    app.text_inserter.insert_text.assert_called_once_with("hello")
    # ...and never touched the swapped-in, already-active transcriber.
    new.transcribe.assert_not_called()
    # cleanup() ran exactly once.
    old.cleanup.assert_called_once()
    # The invariant the test proves: cleanup() ran only after the worker
    # had returned from transcribe() (released the transcriber lock).
    # Recorded inside cleanup() itself, so there is no main-thread check
    # window to race against.
    assert cleanup_saw_worker_done and cleanup_saw_worker_done[0], (
        "cleanup() ran while the worker was still mid-transcribe() — "
        "the transcriber lock is not held across the inference call"
    )
    assert app.transcriber is new


def test_reload_model_serializes_concurrent_reloads(app):
    """Two overlapping model switches must not let a superseded reload
    commit its (stale) transcriber or clean up a transcriber a newer
    reload already made live (issue #22).

    Reload B is spawned while reload A is still in its constructor. B
    swaps in transcriber B. When A's constructor returns, A must detect
    that it was superseded and discard its result — committing it would
    roll back B's swap, and A's cleanup would tear down the live
    transcriber B (or, after B's own cleanup, resurrect nothing on top
    of it).

    Fails without the fix: without reload serialization, A commits its
    stale transcriber A2 and calls cleanup() on the live transcriber B.
    """
    a2 = MagicMock(spec=Transcriber)  # A's constructor result
    b = MagicMock(spec=Transcriber)  # B's constructor result
    original = MagicMock(spec=Transcriber)
    app.transcriber = original

    a_returned = threading.Event()
    a2_built = threading.Event()

    def a_constructor():
        a2_built.set()
        a_returned.wait(timeout=10.0)
        return a2

    a_started = threading.Event()

    def a_reload():
        a_started.set()
        with patch(
            "kuiskaus.model_reload.ParakeetTranscriber", side_effect=a_constructor
        ):
            app._reload_model("parakeet")

    a_thread = threading.Thread(target=a_reload)
    a_thread.start()
    # Wait until A is inside its (blocking) constructor, then start B.
    assert a_started.wait(timeout=5.0), "reload A never started"
    assert a2_built.wait(timeout=5.0)
    with patch("kuiskaus.model_reload.ParakeetTranscriber", return_value=b):
        b_thread = threading.Thread(target=lambda: app._reload_model("parakeet"))
        b_thread.start()
        b_thread.join(timeout=10.0)
    a_returned.set()
    a_thread.join(timeout=10.0)

    # B's swap is the final state; A's stale result was discarded.
    assert app.transcriber is b
    # A's stale constructor result was never committed (it is not the
    # live transcriber) and was released exactly once: a superseded
    # reload must clean up its own (already loaded) model before
    # discarding it, but must never touch the transcriber B made live.
    a2.cleanup.assert_called_once()
    # B's old transcriber (the original) was cleaned up exactly once.
    original.cleanup.assert_called_once()


def test_utcnow_helper_returns_aware_utc():
    """_utcnow() is the single source of the aware-UTC invariant (issue
    #22): aware UTC so it can be subtracted from session_start without
    a TypeError from a naive/other-tz datetime. (No monotonicity
    assertion: the host wall clock steps backward.)"""
    from kuiskaus.menubar import _utcnow

    now = _utcnow()

    assert now.tzinfo is UTC


def test_show_stats_uses_aware_utc_session_start(app, monkeypatch):
    """show_stats() must compute the session duration from aware-UTC
    datetimes (issue #22) so the subtraction cannot raise TypeError."""
    import kuiskaus.menubar as menubar_module

    # Pin the clock 30 minutes after session_start so the rendered
    # duration is deterministic. datetime.datetime is immutable, so the
    # monkeypatch replaces the module-level name with a subclass that
    # forwards everything except now().
    app.session_start = datetime(2026, 1, 1, tzinfo=UTC)
    real_datetime = type(datetime)

    class _PinnedDateTime(real_datetime):  # type: ignore[valid-type, misc]
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, 0, 30, tzinfo=UTC)

    monkeypatch.setattr(menubar_module, "datetime", _PinnedDateTime)

    with patch.object(rumps, "alert") as mock_alert:
        app.show_stats(None)

    mock_alert.assert_called_once()
    assert "0h 30m" in mock_alert.call_args.args[1]


def test_insert_failure_surfaces_via_update_status(app):
    """A transcription whose INSERTION failed must surface the distinct
    🔴 Insert failed banner carrying TextInserter.last_error (#41), not
    a silent success."""
    app.transcriber = MagicMock()
    app.transcriber.transcribe.return_value = {"text": "hello world"}
    app.text_inserter = MagicMock()
    app.text_inserter.insert_text.return_value = False
    app.text_inserter.last_error = "keystroke injection failed"
    app.audio_recorder.last_error = None

    app._transcribe_and_insert(np.array([0.1], dtype=np.float32), 1.0)

    app.text_inserter.insert_text.assert_called_once_with("hello world")
    assert "Insert failed" in app.status_item.title
    assert "keystroke injection failed" in app.status_item.title


def test_insert_failure_status_not_clobbered_by_completion_path(app):
    """The completion path's terminal 🟢 Ready update must be skipped when
    insert_text returns False (#41 no-clobber guard): without it, the
    🔴 Insert failed banner is immediately overwritten and the failure is
    as invisible as the pre-#41 silent no-op."""
    app.transcriber = MagicMock()
    app.transcriber.transcribe.return_value = {"text": "hello world"}
    app.text_inserter = MagicMock()
    app.text_inserter.insert_text.return_value = False
    app.text_inserter.last_error = "keystroke injection failed"
    app.audio_recorder.last_error = None

    app._transcribe_and_insert(np.array([0.1], dtype=np.float32), 1.0)

    assert app.status_item.title == ("🔴 Insert failed: keystroke injection failed")


def test_insert_failure_wins_when_mic_error_also_live(app):
    """A failed insert surfaces its own 🔴 Insert failed banner even when a
    mic error is also live (#41) -- the insert-failure branch is not gated
    on audio_recorder.last_error, so the keystroke/pasteboard failure
    reason is what the user sees."""
    app.transcriber = MagicMock()
    app.transcriber.transcribe.return_value = {"text": "hello world"}
    app.text_inserter = MagicMock()
    app.text_inserter.insert_text.return_value = False
    app.text_inserter.last_error = "keystroke injection failed"
    app.audio_recorder.last_error = "some mic error"

    app._transcribe_and_insert(np.array([0.1], dtype=np.float32), 1.0)

    assert app.status_item.title == ("🔴 Insert failed: keystroke injection failed")


def test_insert_success_still_restores_ready(app):
    """An insert that succeeded still restores the normal Ready state
    (#41) -- the no-clobber guard must not swallow the success path."""
    app.transcriber = MagicMock()
    app.transcriber.transcribe.return_value = {"text": "hello world"}
    app.text_inserter = MagicMock()
    app.text_inserter.insert_text.return_value = True
    app.audio_recorder.last_error = None

    app._transcribe_and_insert(np.array([0.1], dtype=np.float32), 1.0)

    assert app.status_item.title == "🟢 Ready"


def test_app_module_exposes_shared_silicon_check():
    """app.py must use the shared implementation (issue #22 dedup) rather
    than its own private copy."""
    from kuiskaus.app import check_apple_silicon
    from kuiskaus.silicon_check import check_apple_silicon as shared

    assert check_apple_silicon is shared


def test_init_installs_locks_and_transcriber_before_hotkey_listener(
    monkeypatch: pytest.MonkeyPatch,
):
    """Real __init__ ordering (issue #22): the transcriber lock and the
    reload-serialization state must exist before the hotkey listener
    starts, so a worker spawned by the first hotkey already sees the
    locks the reload path serializes on. The hand-built fixture bypasses
    __init__, so this construction test pins the real ordering."""
    _install_stubs(monkeypatch)
    import kuiskaus.menubar as menubar_module

    listener_start = threading.Event()
    original_start = menubar_module.HotkeyListenerCGEvent

    class _TrackingListener(original_start):
        def start(self):
            listener_start.set()
            return original_start.start(self)

    # rumps.App.__init__ needs a display; run it headless-safe via
    # NSApplication is already stubbed-free here (rumps works in tests
    # because it defers the run loop to app.run()). The AudioRecorder
    # stub swallows the on_capture_started kwarg (issue #43); a MagicMock
    # would leak a partially-constructed recorder from __del__ ->
    # cleanup() otherwise.
    with (
        patch.object(menubar_module, "HotkeyListenerCGEvent", _TrackingListener),
        patch.object(menubar_module, "AudioRecorder") as mock_recorder_cls,
    ):
        app = menubar_module.KuiskausMenuBarApp()

    mock_recorder_cls.assert_called_once_with(
        on_capture_started=app._enqueue_capture_started
    )

    # The listener has been started (synchronously in __init__), so the
    # ordering invariant is fully exercised: the lock, the reload
    # serialization state, and the transcriber protocol guard are all in
    # place before the listener's start() returned.
    assert listener_start.is_set()
    assert isinstance(app._transcriber_lock, type(threading.Lock()))
    assert hasattr(app, "_reload_lock")
    assert app._reload_generation == 0
    assert isinstance(app.transcriber, Transcriber)


def test_toggle_keep_warm_flips_state_and_tells_recorder(app):
    item = rumps.MenuItem("Keep mic warm", callback=None)
    app.keep_warm = False

    app.toggle_keep_warm(item)
    assert app.keep_warm is True
    assert item.state == 1
    app.audio_recorder.set_keep_warm.assert_called_with(True)

    app.toggle_keep_warm(item)
    assert app.keep_warm is False
    assert item.state == 0
    app.audio_recorder.set_keep_warm.assert_called_with(False)
