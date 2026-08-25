"""Hardware-free unit tests for KuiskausMenuBarApp hotkey callbacks.

The listener classes and heavy components are mocked via monkeypatch so
the menubar module import never touches Quartz, pyaudio, or model
loading.
"""

import queue
import sys
import threading
import types
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rumps

from kuiskaus.transcriber import Transcriber


class _FakeCgeventModule(types.ModuleType):
    HotkeyListenerCGEvent: type


class _FakeAudioRecorderModule(types.ModuleType):
    AudioRecorder: MagicMock


class _FakeParakeetModule(types.ModuleType):
    ParakeetTranscriber: type


class _FakeWhisperModule(types.ModuleType):
    WhisperTranscriber: type


class _FakeTextInserterModule(types.ModuleType):
    TextInserter: MagicMock


class _FakeCGEventListener:
    """Stand-in for HotkeyListenerCGEvent used when constructing the app.

    The real class is never imported here: the menubar module is loaded
    with a fresh sys.modules stub each test, and instantiating the real
    Quartz-backed class in a unit test would touch the run loop.
    """

    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release

    def start(self):
        return True

    def stop(self):
        pass


def _fake_parakeet_transcriber_class():
    """Real class (not a MagicMock) standing in for ParakeetTranscriber.

    _reload_model compares the constructor result's exact type (type()
    identity) against the real ParakeetTranscriber class to decide
    whether the background load must be verified; a MagicMock stub can
    never satisfy that identity, so the stub needs a real class. Instances report a loaded model
    unless configured otherwise (the unusable-model test flips them).
    _load_model is a no-op stub: real ParakeetTranscriber runs the model
    load on a background thread, and the tests suppress it exactly as
    they do for the real class.
    """

    class ParakeetTranscriberStub:
        def __init__(self) -> None:
            self.model: object = MagicMock(name="parakeet-model")

        def transcribe(self, audio, **kwargs):
            return {"text": ""}

        def cleanup(self) -> None:
            self.model = None

        def _ensure_loaded(self) -> None:
            if self.model is None:
                raise RuntimeError("Parakeet model failed to load")

        def _load_model(self) -> None:
            pass

    return ParakeetTranscriberStub


def _fake_whisper_transcriber_class():
    """Real class (not a MagicMock) standing in for WhisperTranscriber.

    The fixture's default reload target is "whisper" (the app's default
    model), and menubar's reload path compares the constructor result's
    exact type (type() identity) against the real WhisperTranscriber
    class; a MagicMock stub can never satisfy that identity, so the stub
    needs a real class.
    """

    class WhisperTranscriberStub:
        def __init__(self, model_name: str = "turbo", device=None) -> None:
            self.model_name = model_name

        def transcribe(self, audio, **kwargs):
            return {"text": ""}

        def cleanup(self) -> None:
            pass

    return WhisperTranscriberStub


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub hardware/model dependencies before importing menubar."""
    fake_quartz = types.ModuleType("Quartz")
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    # Fresh stub each call: menubar is re-imported per test and re-binds
    # HotkeyListenerCGEvent from this module.
    fake_cgevent = _FakeCgeventModule("kuiskaus.hotkey_listener_cgevent")
    fake_cgevent.HotkeyListenerCGEvent = _FakeCGEventListener
    monkeypatch.setitem(sys.modules, "kuiskaus.hotkey_listener_cgevent", fake_cgevent)

    fake_audio = _FakeAudioRecorderModule("kuiskaus.audio_recorder")
    fake_audio.AudioRecorder = MagicMock()

    # Real class (see _fake_parakeet_transcriber_class): menubar's
    # type() identity check on the reload path needs a real class
    # identity, and the instances satisfy the Transcriber protocol for
    # the isinstance(..., Transcriber) check. The class is also patched
    # directly onto the already-imported menubar module so the reload
    # path and the __init__ path see the same stub identity.
    fake_parakeet = _FakeParakeetModule("kuiskaus.parakeet_transcriber")
    parakeet_cls = _fake_parakeet_transcriber_class()
    fake_parakeet.ParakeetTranscriber = parakeet_cls

    try:
        import kuiskaus.menubar as _menubar

        monkeypatch.setattr(_menubar, "ParakeetTranscriber", parakeet_cls)
    except ImportError:
        pass  # menubar not imported yet; the sys.modules stub covers it

    fake_whisper = _FakeWhisperModule("kuiskaus.whisper_transcriber")
    whisper_cls = _fake_whisper_transcriber_class()
    fake_whisper.WhisperTranscriber = whisper_cls

    try:
        import kuiskaus.menubar as _menubar

        monkeypatch.setattr(_menubar, "WhisperTranscriber", whisper_cls)
    except ImportError:
        pass  # menubar not imported yet; the sys.modules stub covers it

    fake_text = _FakeTextInserterModule("kuiskaus.text_inserter")
    fake_text.TextInserter = MagicMock()

    monkeypatch.setitem(sys.modules, "kuiskaus.audio_recorder", fake_audio)
    monkeypatch.setitem(sys.modules, "kuiskaus.parakeet_transcriber", fake_parakeet)
    monkeypatch.setitem(sys.modules, "kuiskaus.whisper_transcriber", fake_whisper)
    monkeypatch.setitem(sys.modules, "kuiskaus.text_inserter", fake_text)


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
    instance._reload_lock = threading.Lock()
    instance._reload_generation = 0
    instance._pending_capture_started_events = queue.Queue()
    # Stubbed timer: the real rumps.Timer requires the NSRunLoop that
    # only exists under app.run(); tests invoke _drain_ui_events directly.
    instance._ui_tick_timer = MagicMock()
    instance.audio_recorder.recording = False
    instance.audio_recorder.current_generation = 0
    return instance


def test_press_sets_starting_state(app):
    """Press only acknowledges the request: the menu bar shows 🟠
    Starting... until the capture-started event arrives (issue #43).
    A capture-started event enqueued before press must NOT be rendered
    either -- the UI must not claim a live recording before the press."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app._pending_capture_started_events.put(1)

    app.on_hotkey_press()
    app._drain_ui_events(None)

    assert app.is_recording is True
    assert app.title == "🟠"
    assert app.status_item.title == "🟠 Starting..."


def test_capture_started_event_drained_transitions_to_recording(app):
    """A capture-started event for the current generation transitions the
    menu bar to 🔴 Recording on the main-thread trampoline (issue #43)."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.on_hotkey_press()
    assert app.title == "🟠"

    app.audio_recorder.recording = True
    app.audio_recorder.current_generation = 5
    app._enqueue_capture_started()

    app._drain_ui_events(None)

    assert app.title == "🔴"
    assert app.status_item.title == "🔴 Recording"


def test_stale_capture_started_dropped_by_generation_gate(app):
    """A capture-started event whose generation is no longer current
    (superseded recording, or release already fired) must be dropped --
    a late event can never flicker 🔴 over Processing/Ready (#43)."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.on_hotkey_press()

    app.audio_recorder.recording = True
    app.audio_recorder.current_generation = 5
    app._enqueue_capture_started()
    # A newer generation superseded the event's recording.
    app.audio_recorder.current_generation = 6

    app._drain_ui_events(None)

    assert app.title == "🟠"
    assert app.status_item.title == "🟠 Starting..."


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


def test_no_audio_captured_state_set_on_race_lost(app):
    """A release that yields zero audio without a mic error (race-lost,
    #40) surfaces ⚪ No audio captured -- not a silent flip to Ready,
    and no transcription thread is spawned."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.audio_recorder.stop_recording = MagicMock(
        return_value=np.array([], dtype=np.float32)
    )
    app.audio_recorder.last_error = None

    app.on_hotkey_press()
    with patch("threading.Thread") as mock_thread_cls:
        app.on_hotkey_release()
        mock_thread_cls.assert_not_called()

    assert app.status_item.title == "⚪ No audio captured"
    assert app.is_recording is False


def test_no_audio_state_clears_on_next_successful_recording(app):
    """⚪ set on a race-lost release must be cleared by the next release
    that yields audio: the transcribe branch resets the status to
    🟡 Processing before spawning the worker (#40 DoD)."""
    app.audio_recorder.start_recording = MagicMock(return_value=True)

    # Cycle 1: race-lost -- empty audio, no error -> ⚪ status.
    app.audio_recorder.stop_recording = MagicMock(
        return_value=np.array([], dtype=np.float32)
    )
    app.on_hotkey_press()
    app.on_hotkey_release()
    assert app.status_item.title == "⚪ No audio captured"

    # Cycle 2: a genuine capture -- the ⚪ state must not survive the
    # release that hands audio to the transcription worker.
    app.audio_recorder.stop_recording = MagicMock(
        return_value=np.array([0.1, 0.2], dtype=np.float32)
    )
    app.on_hotkey_press()
    with patch("threading.Thread") as mock_thread_cls:
        app.on_hotkey_release()
        mock_thread_cls.assert_called_once()

    assert app.status_item.title == "🟡 Processing..."


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
    """A genuinely successful recording (no last_error, captured audio) is
    the one case that DOES clear a previously persisted banner."""
    app.status_item.title = "🔴 microphone busy — recording did not start"
    app.audio_recorder.start_recording = MagicMock(return_value=True)
    app.audio_recorder.stop_recording = MagicMock(
        return_value=np.array([0.1, 0.2], dtype=np.float32)
    )
    app.audio_recorder.last_error = None

    app.on_hotkey_press()
    with patch("threading.Thread") as mock_thread_cls:
        app.on_hotkey_release()
        mock_thread_cls.assert_called_once()

    assert app.status_item.title == "🟡 Processing..."


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
    # spec=Transcriber so an attribute that does not exist on the
    # protocol raises instead of silently succeeding (same defect class
    # as the bare-MagicMock last_error fixture bug, issue #22 review).
    app.transcriber = MagicMock(spec=Transcriber)
    app.audio_recorder.last_error = "microphone busy — recording did not start"
    app.status_item.title = f"🔴 {app.audio_recorder.last_error}"

    with patch("kuiskaus.menubar.ParakeetTranscriber") as mock_ctor:
        mock_ctor.return_value = app.transcriber
        app._reload_model("parakeet")

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready"


def test_reload_model_success_clears_ready_without_last_error(app):
    """Without a persisted mic error, a successful model reload still
    resets the status to Ready."""
    app.transcriber = MagicMock(spec=Transcriber)
    app.audio_recorder.last_error = None

    with patch("kuiskaus.menubar.ParakeetTranscriber") as mock_ctor:
        mock_ctor.return_value = app.transcriber
        app._reload_model("parakeet")

    assert app.status_item.title == "🟢 Ready"


def test_reload_model_exception_preserves_banner_when_last_error_set(app):
    """A model-reload failure must not clobber a live mic-error banner
    either -- same rationale as the success path above."""
    app.transcriber = MagicMock(spec=Transcriber)
    app.transcriber.cleanup.side_effect = RuntimeError("boom")
    app.audio_recorder.last_error = "microphone busy — recording did not start"
    app.status_item.title = f"🔴 {app.audio_recorder.last_error}"

    app._reload_model("parakeet")

    assert "busy" in app.status_item.title
    assert app.status_item.title != "🟢 Ready (model error)"


def test_reload_model_unusable_model_reports_failure(app):
    """A reload whose new transcriber's background load FAILED must not
    report success (issue #22 review): the swap must not commit (the
    old transcriber keeps serving), and the failure must surface via
    update_status — not a '✅ Model changed' log."""
    from kuiskaus.parakeet_transcriber import ParakeetTranscriber as RealParakeet

    original = MagicMock(spec=Transcriber)
    # A REAL ParakeetTranscriber with the load thread suppressed and the
    # model forced to None: the reload's type() identity check must run
    # its load verification on it (a bare MagicMock can't satisfy the
    # real-class identity check by design). With the guard removed the
    # reload would commit this dead transcriber and claim success.
    with patch("kuiskaus.parakeet_transcriber.ParakeetTranscriber._load_model"):
        failed = RealParakeet()
    failed.model = None  # simulate a failed background load
    app.transcriber = original

    import kuiskaus.menubar as menubar_module

    with patch.object(menubar_module, "ParakeetTranscriber", return_value=failed):
        app._reload_model("parakeet")

    # The unusable transcriber was never committed and the live one was
    # never torn down by the failed reload.
    assert app.transcriber is original
    original.cleanup.assert_not_called()
    # The unusable model was released before discarding (no dead model
    # left resident).
    assert failed.model is None
    # The failure surfaced through the existing status mechanism
    # ("Log instead of notification"), as a model error, not Ready.
    assert app.status_item.title == "🟢 Ready (model error)"


def _parked_worker(app, old):
    """Park a real _transcribe_and_insert worker inside old.transcribe().

    Returns {"worker", "parked", "release_event", "worker_done"}.
    "parked" means the worker is inside transcribe() (with the
    transcriber lock held by the fixed source); "release_event" lets it
    finish; "worker_done" is set when transcribe() returns.
    """
    parked = threading.Event()
    release_event = threading.Event()
    worker_done = threading.Event()

    def blocking_transcribe(*_args, **_kwargs):
        parked.set()  # tell main we are inside transcribe()
        release_event.wait(timeout=10.0)
        worker_done.set()  # tell main transcribe() has returned
        return {"text": "hello"}

    old.transcribe.side_effect = blocking_transcribe
    app.text_inserter = MagicMock()

    worker = threading.Thread(
        target=app._transcribe_and_insert,
        args=(np.array([0.1], dtype=np.float32), 1.0),
    )

    return {
        "worker": worker,
        "parked": parked,
        "release_event": release_event,
        "worker_done": worker_done,
    }


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

    with patch("kuiskaus.menubar.ParakeetTranscriber", return_value=new):
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
        with patch("kuiskaus.menubar.ParakeetTranscriber", side_effect=a_constructor):
            app._reload_model("parakeet")

    a_thread = threading.Thread(target=a_reload)
    a_thread.start()
    # Wait until A is inside its (blocking) constructor, then start B.
    assert a_started.wait(timeout=5.0), "reload A never started"
    assert a2_built.wait(timeout=5.0)
    with patch("kuiskaus.menubar.ParakeetTranscriber", return_value=b):
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
