"""Hardware-free tests for the #17 hotkey ordering guarantees.

Covers the Definition of Done for issue #17:
- both listeners route on_press/on_release through one shared
  single-threaded executor (callbacks run in event order);
- a release whose press was never seen is a no-op:
  menu bar keeps the Ready status and the CLI does not call the
  recorder at all.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from kuiskaus.hotkey_dispatcher import HotkeyDispatcher

# conftest.py pre-mocks the macOS app/event layer (Quartz, AppKit,
# PyObjCTools.AppHelper, ApplicationServices) before these imports.
from kuiskaus.hotkey_listener import HotkeyListener
from kuiskaus.hotkey_listener_cgevent import HotkeyListenerCGEvent

_FLAGS_ON = (1 << 18) | (1 << 19)  # Control + Option modifier masks


class TestHotkeyDispatcher:
    def test_callbacks_run_in_submission_order(self):
        dispatcher = HotkeyDispatcher()
        try:
            order: list[str] = []
            for name in ("press", "release", "press2", "release2"):
                dispatcher.dispatch(lambda n=name: order.append(n))
            deadline = time.time() + 5
            while len(order) < 4 and time.time() < deadline:
                time.sleep(0.01)
            assert order == ["press", "release", "press2", "release2"]
        finally:
            dispatcher.shutdown()

    def test_dispatch_serializes_callbacks_across_threads(self):
        dispatcher = HotkeyDispatcher()
        try:
            events: list[int] = []
            events_lock = threading.Lock()
            started = threading.Barrier(21)

            def producer(i: int) -> None:
                started.wait()
                with events_lock:
                    events.append(i)

            for i in range(20):
                threading.Thread(target=producer, args=(i,)).start()
            # Submit each callback only after its producer thread is fully
            # alive and recorded: the worker must then execute the 20
            # callbacks in this exact submission order.
            started.wait()
            for i in range(20):
                deadline = time.time() + 5
                while True:
                    with events_lock:
                        ready = i in events
                    if ready or time.time() > deadline:
                        break
                    time.sleep(0.001)
                assert ready, f"producer {i} did not run"
                dispatcher.dispatch(lambda i=i: events.append(1000 + i))
            deadline = time.time() + 5
            while len(events) < 40 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            dispatcher.shutdown()
        # The 20 dispatched callbacks form a contiguous, in-order tail:
        # each ran only after its producer thread recorded itself.
        assert events[-20:] == list(range(1000, 1020))

    def test_shutdown_stops_worker(self):
        dispatcher = HotkeyDispatcher()
        dispatcher.shutdown()
        deadline = time.time() + 5
        while dispatcher._thread.is_alive() and time.time() < deadline:
            time.sleep(0.01)
        assert not dispatcher._thread.is_alive()


class TestListenerDispatcherWiring:
    @pytest.mark.parametrize(
        "make_listener",
        [
            lambda: HotkeyListener(on_press=lambda: None, on_release=lambda: None),
            lambda: HotkeyListenerCGEvent(
                on_press=lambda: None, on_release=lambda: None
            ),
        ],
    )
    def test_listeners_use_one_shared_dispatcher(self, make_listener):
        listener = make_listener()
        try:
            assert isinstance(listener.dispatcher, HotkeyDispatcher)
            assert listener.dispatcher._thread.is_alive()
            # Callbacks run on the dedicated worker, in order.
            done = []
            listener.dispatcher.dispatch(lambda: done.append(1))
            deadline = time.time() + 5
            while not done and time.time() < deadline:
                time.sleep(0.01)
            assert done == [1]
            assert not threading.current_thread() is listener.dispatcher._thread
        finally:
            listener.dispatcher.shutdown()


class TestMenubarReleaseNoop:
    def _make_app(self):
        import kuiskaus.menubar as menubar_module

        app = menubar_module.KuiskausMenuBarApp.__new__(
            menubar_module.KuiskausMenuBarApp
        )
        app.enabled = True
        app.is_recording = False
        app.recording_start_time = None
        statuses: list[str] = []
        app.update_status = lambda status: statuses.append(status)
        app.audio_recorder = MagicMock()
        return app, statuses

    def test_release_without_press_is_noop_and_keeps_ready(self):
        app, statuses = self._make_app()
        app.on_hotkey_release()
        # No recorder interaction and no status change (already Ready).
        app.audio_recorder.stop_recording.assert_not_called()
        app.audio_recorder.start_recording.assert_not_called()
        assert statuses == []


class TestCliReleaseNoop:
    def _make_app(self):
        import kuiskaus.app as app_module

        app = app_module.KuiskausApp.__new__(app_module.KuiskausApp)
        app.is_recording = False
        app.recording_start_time = None
        app.audio_recorder = MagicMock()
        app.show_notification = lambda *a, **k: None
        return app

    def test_release_without_prints_noop(self, capsys):
        app = self._make_app()
        app.on_hotkey_release()
        app.audio_recorder.stop_recording.assert_not_called()
        app.audio_recorder.start_recording.assert_not_called()
        assert "ignoring release" in capsys.readouterr().out

    def test_press_press_release_records_exactly_once(self):
        app = self._make_app()
        app.audio_recorder.stop_recording.return_value = []
        app.on_hotkey_press()
        app.on_hotkey_press()  # second press while recording: ignored
        app.on_hotkey_release()
        app.audio_recorder.start_recording.assert_called_once()
        app.audio_recorder.stop_recording.assert_called_once()
        assert app.is_recording is False


class _FakeNSEvent:
    """Fake NSEvent matching NSFlagsChangedMask + modifierFlags."""

    NSFlagsChangedMask = 1 << 12

    def __init__(self, pressed: bool):
        self._pressed = pressed

    def type(self):
        return self.NSFlagsChangedMask

    def modifierFlags(self):
        return _FLAGS_ON if self._pressed else 0


def _drive_press_release(listener, pressed_states):
    """Feed modifier transitions in event order, then drain the dispatcher.

    Simulates the listener's press/release transitions (both gate on
    ``is_pressed`` and dispatch through the shared worker) without a
    macOS event loop.  The AppKit/Quartz layers are pre-mocked in
    conftest, so the mocked module constants are set to real values.
    """
    from AppKit import NSEvent
    from Quartz import kCGEventFlagsChanged

    NSEvent.NSFlagsChangedMask = _FakeNSEvent.NSFlagsChangedMask

    if isinstance(listener, HotkeyListenerCGEvent):
        for pressed in pressed_states:
            with patch(
                "Quartz.CGEventGetFlags",
                return_value=_FLAGS_ON if pressed else 0,
            ):
                listener._event_tap_callback(
                    None, kCGEventFlagsChanged, None, None
                )
    else:
        for pressed in pressed_states:
            listener._handle_event(_FakeNSEvent(pressed))

    deadline = time.time() + 5
    while not listener.dispatcher._queue.empty() and time.time() < deadline:
        time.sleep(0.01)


class _FakeHotkeyListener:
    """Listener-shaped object mirroring both real listeners' dispatch
    logic (gate on is_pressed, dispatch through the shared worker)."""

    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.is_pressed = False
        self.dispatcher = HotkeyDispatcher()

    def _handle_event(self, event):
        if event.type() != _FakeNSEvent.NSFlagsChangedMask:
            return event
        modifiers_pressed = event.modifierFlags() != 0
        if modifiers_pressed and not self.is_pressed:
            self.is_pressed = True
            if self.on_press:
                self.dispatcher.dispatch(self.on_press)
        elif not modifiers_pressed and self.is_pressed:
            self.is_pressed = False
            if self.on_release:
                self.dispatcher.dispatch(self.on_release)
        return event


class TestListenerEventOrdering:
    def _make_cli_app(self):
        import kuiskaus.app as app_module

        app = app_module.KuiskausApp.__new__(app_module.KuiskausApp)
        app.is_recording = False
        app.recording_start_time = None
        app.audio_recorder = MagicMock()
        app.show_notification = lambda *a, **k: None
        return app

    @pytest.mark.parametrize(
        "make_listener",
        [
            lambda app: HotkeyListener(
                on_press=app.on_hotkey_press, on_release=app.on_hotkey_release
            ),
            lambda app: HotkeyListenerCGEvent(
                on_press=app.on_hotkey_press, on_release=app.on_hotkey_release
            ),
            lambda app: _FakeHotkeyListener(
                app.on_hotkey_press, app.on_hotkey_release
            ),
        ],
    )
    def test_press_release_records_exactly_once(self, make_listener):
        app = self._make_cli_app()
        app.audio_recorder.stop_recording.return_value = []
        listener = make_listener(app)
        try:
            _drive_press_release(listener, [True, False])
        finally:
            listener.dispatcher.shutdown()
        app.audio_recorder.start_recording.assert_called_once()
        app.audio_recorder.stop_recording.assert_called_once()
        assert app.is_recording is False

    @pytest.mark.parametrize(
        "make_listener",
        [
            lambda app: HotkeyListener(
                on_press=app.on_hotkey_press, on_release=app.on_hotkey_release
            ),
            lambda app: HotkeyListenerCGEvent(
                on_press=app.on_hotkey_press, on_release=app.on_hotkey_release
            ),
            lambda app: _FakeHotkeyListener(
                app.on_hotkey_press, app.on_hotkey_release
            ),
        ],
    )
    def test_release_with_no_prior_press_never_touches_recorder(self, make_listener):
        app = self._make_cli_app()
        listener = make_listener(app)
        try:
            _drive_press_release(listener, [False])
        finally:
            listener.dispatcher.shutdown()
        app.audio_recorder.stop_recording.assert_not_called()
        app.audio_recorder.start_recording.assert_not_called()
        assert app.is_recording is False

    def test_release_before_press_ordering(self):
        """A release arriving before any press is a no-op; the following
        press-release pair then records exactly once, in order."""
        app = self._make_cli_app()
        app.audio_recorder.stop_recording.return_value = []
        listener = _FakeHotkeyListener(
            app.on_hotkey_press, app.on_hotkey_release
        )
        try:
            _drive_press_release(listener, [False, True, False])
        finally:
            listener.dispatcher.shutdown()
        app.audio_recorder.start_recording.assert_called_once()
        app.audio_recorder.stop_recording.assert_called_once()
        assert app.is_recording is False
