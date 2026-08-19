"""Unit tests for the CGEvent hotkey listener's callback dispatcher.

Hardware-free: the Quartz / AppKit / Foundation / PyObjCTools modules are
mocked before ``kuiskaus`` is imported, so no CGEvent tap is created and no
accessibility permission is needed. The mocks are always installed — on CI
machines without pyobjc a real import would fail, and on dev machines the
real modules must not be used (no accessibility permission, no event tap).

Issue #17: press and release callbacks previously ran on independent,
unsequenced threads. They must now run in event order on a single-threaded
dispatcher so a release always observes the state of its press.
"""

import sys
import threading
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock the native macOS modules before importing the listener under test.
# ---------------------------------------------------------------------------

if "Quartz" not in sys.modules:
    quartz = types.ModuleType("Quartz")
    quartz.kCGEventFlagsChanged = 12
    quartz.CGEventGetFlags = MagicMock(return_value=0)
    quartz.CFRunLoopRun = MagicMock()
    quartz.CFRunLoopStop = MagicMock()
    quartz.CFRunLoopAddSource = MagicMock()
    quartz.CFRunLoopRemoveSource = MagicMock()
    quartz.CFRunLoopGetCurrent = MagicMock()
    quartz.CFMachPortCreateRunLoopSource = MagicMock()
    quartz.CGEventTapCreate = MagicMock()
    quartz.CGEventTapEnable = MagicMock()
    quartz.CGEventMaskBit = MagicMock()
    quartz.kCGSessionEventTap = 0
    quartz.kCGHeadInsertEventTap = 0
    quartz.kCGEventTapOptionListenOnly = 0
    quartz.kCFRunLoopDefaultMode = "kCFRunLoopDefaultMode"
    sys.modules["Quartz"] = quartz

if "AppKit" not in sys.modules:
    appkit = types.ModuleType("AppKit")
    appkit.__getattr__ = lambda name: MagicMock(name=f"AppKit.{name}")
    appkit.NSPasteboardTypeString = "public.utf8-plain-text"
    sys.modules["AppKit"] = appkit

if "ApplicationServices" not in sys.modules:
    app_services = types.ModuleType("ApplicationServices")
    app_services.AXIsProcessTrusted = MagicMock(return_value=True)
    sys.modules["ApplicationServices"] = app_services

if "PyObjCTools" not in sys.modules:
    pyobjc_tools_pkg = types.ModuleType("PyObjCTools")
    pyobjc_tools_pkg.__path__ = []
    pyobjc_tools_app_helper = types.ModuleType("PyObjCTools.AppHelper")
    pyobjc_tools_app_helper.InstallMenuItem = MagicMock()
    pyobjc_tools_app_helper.CallFunctionLater = MagicMock()
    pyobjc_tools_app_helper.NextEvent = MagicMock()
    pyobjc_tools_app_helper.MainLoop = MagicMock()
    pyobjc_tools_app_helper.ExitMainLoop = MagicMock()
    pyobjc_tools_pkg.AppHelper = pyobjc_tools_app_helper
    sys.modules["PyObjCTools.AppHelper"] = pyobjc_tools_app_helper
    sys.modules["PyObjCTools"] = pyobjc_tools_pkg

if "Foundation" not in sys.modules:
    foundation = types.ModuleType("Foundation")
    foundation.__getattr__ = lambda name: MagicMock(name=f"Foundation.{name}")
    foundation.NSUserNotification = MagicMock()
    foundation.NSUserNotificationCenter = MagicMock()
    sys.modules["Foundation"] = foundation

# rumps pulls Foundation in at import time; keep it off the import path.
if "rumps" not in sys.modules:
    rumps_mock = types.ModuleType("rumps")
    rumps_mock.__getattr__ = lambda name: MagicMock(name=f"rumps.{name}")

    class _FakeApp:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeMenuItem:
        def __init__(self, *args, **kwargs):
            pass

    rumps_mock.App = _FakeApp
    rumps_mock.MenuItem = _FakeMenuItem
    rumps_mock.separator = None
    rumps_mock.clicked = lambda *a, **kw: lambda fn: fn
    rumps_mock.alert = MagicMock()
    rumps_mock.quit_application = MagicMock()
    rumps_mock.timers = MagicMock()
    sys.modules["rumps"] = rumps_mock


from datetime import UTC

from kuiskaus.hotkey_listener_cgevent import HotkeyListenerCGEvent

KCGEVENTFLAGSCHANGED = 12

CONTROL = 1 << 18
OPTION = 1 << 19
COMMAND = 1 << 20


def make_listener(on_press, on_release):
    """Build a listener whose event tap is a MagicMock (never real hardware)."""
    return HotkeyListenerCGEvent(on_press=on_press, on_release=on_release)


def deliver_flags(listener, flags):
    """Deliver a kCGEventFlagsChanged event with a specific flag value."""
    with patch(
        "kuiskaus.hotkey_listener_cgevent.Quartz.CGEventGetFlags",
        return_value=flags,
    ):
        listener._event_tap_callback(None, KCGEVENTFLAGSCHANGED, MagicMock(), None)


# ---------------------------------------------------------------------------
# Dispatcher ordering
# ---------------------------------------------------------------------------


class TestDispatcherOrdering:
    def test_release_before_press_runs_in_event_order(self):
        """A fast press-release must run on_press before on_release, even
        though both are dispatched before either worker iteration completes."""
        order = []

        def on_press():
            # Block until the release callback has also been dispatched so
            # both are pending; with unsequenced threads the release would be
            # free to run first.
            release_dispatched.wait(timeout=5)
            order.append("press")

        def on_release():
            order.append("release")

        listener = make_listener(on_press, on_release)
        release_dispatched = threading.Event()
        original_tap_callback = listener._event_tap_callback

        def tap_callback(*args):
            result = original_tap_callback(*args)
            # The release event is the second call; flag it so the blocked
            # press callback can finish.
            if not listener.is_pressed:
                release_dispatched.set()
            return result

        listener._event_tap_callback = tap_callback
        listener.start()
        try:
            # Deliver a press, then a release without letting the press
            # callback finish (it blocks on release_dispatched).
            deliver_flags(listener, CONTROL | OPTION)
            # is_pressed is set synchronously in the tap callback, so the
            # release event is now recognised.
            assert listener.is_pressed is True
            deliver_flags(listener, 0)
            listener._drain_for_tests()
        finally:
            listener.stop()

        # Both callbacks were pending simultaneously; the single worker
        # guarantees press ran (and completed) before release did.
        assert order == ["press", "release"]

    def test_press_press_release_runs_in_event_order(self):
        """Rapid events (press, press, release) must be consumed FIFO."""
        order = []
        events = []
        done = threading.Event()

        def on_press():
            events.append("press")
            if len(events) == 2:
                done.set()

        def on_release():
            order.append("release")

        listener = make_listener(on_press, on_release)
        listener.start()
        try:
            # First press comes through the tap callback.
            deliver_flags(listener, CONTROL | OPTION)
            # The tap suppresses a second press while is_pressed is True,
            # so simulate the "press, press" stress by dispatching directly
            # in event order through the same dispatcher the tap uses.
            listener._dispatch(on_press)
            deliver_flags(listener, 0)
            assert done.wait(timeout=5)
            listener._drain_for_tests()
        finally:
            listener.stop()

        assert events == ["press", "press"]
        assert order == ["release"]

    def test_tap_callback_does_not_run_callbacks_inline(self):
        """Delivering an event must return before the callback runs: the
        tap callback enqueues and never executes callbacks inline."""
        called = threading.Event()

        def on_press():
            called.set()

        def on_release():
            pass

        listener = make_listener(on_press, on_release)
        listener.start()
        try:
            deliver_flags(listener, CONTROL | OPTION)
            # The tap callback has returned; if callbacks ran inline this
            # would already be set.
            assert not called.is_set()
            listener._drain_for_tests()
        finally:
            listener.stop()

        assert called.is_set()


# ---------------------------------------------------------------------------
# on_hotkey_release no-op guards (menubar + CLI)
# ---------------------------------------------------------------------------


def _build_menubar_app():
    """Build a KuiskausMenuBarApp without running rumps/PyObjC init."""
    from kuiskaus.menubar import KuiskausMenuBarApp

    app = KuiskausMenuBarApp.__new__(KuiskausMenuBarApp)
    from kuiskaus.audio_recorder import AudioRecorder
    from kuiskaus.text_inserter import TextInserter

    app.audio_recorder = AudioRecorder()
    app.audio_recorder.recording = False
    app.audio_recorder.pyaudio = MagicMock()
    app.transcriber = object()
    app.text_inserter = TextInserter()
    app.is_recording = False
    app.recording_start_time = None
    app.enabled = True
    app.use_apfel = False
    app._apfel_lock = threading.Lock()
    app.total_transcriptions = 0
    app.total_recording_time = 0.0
    from datetime import datetime

    app.session_start = datetime.now(UTC)

    class _StatusItem:
        def __init__(self):
            self.title = "🟢 Ready"

    app.status_item = _StatusItem()
    return app


def _build_cli_app():
    """Build a KuiskausApp without constructing real transcribers."""
    from kuiskaus.app import KuiskausApp
    from kuiskaus.audio_recorder import AudioRecorder
    from kuiskaus.hotkey_listener import HotkeyListener
    from kuiskaus.text_inserter import TextInserter

    with (
        patch.object(AudioRecorder, "__init__", lambda self: None),
        patch.object(TextInserter, "__init__", lambda self: None),
        patch.object(HotkeyListener, "__init__", lambda self, **kw: None),
        patch("kuiskaus.app.ParakeetTranscriber", return_value=object()),
    ):
        app = KuiskausApp.__new__(KuiskausApp)
        app.audio_recorder = AudioRecorder()
        app.audio_recorder.recording = False
        app.audio_recorder.pyaudio = MagicMock()
        app.transcriber = object()
        app.text_inserter = TextInserter()
        app.is_recording = False
        app.recording_start_time = None
        app.use_apfel = False
        app.hotkey_listener = HotkeyListener(
            on_press=app.on_hotkey_press, on_release=app.on_hotkey_release
        )
        app.total_transcriptions = 0
        app.total_recording_time = 0.0
        return app


class TestMenubarReleaseNoOp:
    def test_release_without_recording_is_noop_and_restores_ready(self):
        app = _build_menubar_app()
        app.audio_recorder.stop_recording = MagicMock()

        app.on_hotkey_release()

        app.audio_recorder.stop_recording.assert_not_called()
        assert app.status_item.title == "🟢 Ready"
        assert app.is_recording is False

    def test_release_without_recording_when_disabled_is_noop(self):
        """Even with enabled=False the no-op path must not touch the
        recorder."""
        app = _build_menubar_app()
        app.enabled = False
        app.audio_recorder.stop_recording = MagicMock()

        app.on_hotkey_release()

        app.audio_recorder.stop_recording.assert_not_called()
        assert app.status_item.title == "🟢 Ready"


class TestCLIReleaseNoOp:
    def test_release_without_recording_is_noop(self, capsys):
        app = _build_cli_app()
        app.audio_recorder.stop_recording = MagicMock()

        app.on_hotkey_release()

        app.audio_recorder.stop_recording.assert_not_called()
        assert app.is_recording is False
        out = capsys.readouterr().out
        assert "no active recording" in out.lower()


# ---------------------------------------------------------------------------
# NSEvent (CLI) listener dispatcher — issue #17 requires both listeners
# ---------------------------------------------------------------------------


def _build_nsevent_listener(on_press, on_release):
    """Build an NSEvent HotkeyListener with mocked event monitors."""
    from kuiskaus.hotkey_listener import HotkeyListener

    # The AppKit NSEvent class in this test session may be a MagicMock
    # (installed by test_hotkey_cgevent.py). Patch its monitor methods
    # so start() succeeds without real hardware.
    import kuiskaus.hotkey_listener as hl

    with (
        patch.object(
            hl.NSEvent,
            "addLocalMonitorForEventsMatchingMask_handler_",
            return_value=MagicMock(name="local_monitor"),
        ),
        patch.object(
            hl.NSEvent,
            "addGlobalMonitorForEventsMatchingMask_handler_",
            return_value=MagicMock(name="global_monitor"),
        ),
        patch.object(hl.NSEvent, "removeMonitor_", return_value=None),
    ):
        return HotkeyListener(on_press=on_press, on_release=on_release)


def deliver_flags_nsevent(listener, flags):
    """Deliver an NSFlagsChanged event with a specific modifier flag value."""
    event = MagicMock(name="event")
    event.type.return_value = 4096  # NSFlagsChangedMask = 1 << 12
    event.modifierFlags.return_value = flags
    listener._handle_event(event)


class TestNSEventDispatcherOrdering:
    def test_release_before_press_runs_in_event_order(self, capsys):
        """A fast press-release on the NSEvent (CLI) listener must run
        on_press before on_release even though both are dispatched before
        the worker iteration completes — the race from issue #17."""
        order = []
        release_dispatched = threading.Event()

        def on_press():
            # Block until the release callback has also been dispatched so
            # both are pending; with unsequenced threads the release would
            # be free to run first (and would observe is_recording=False).
            release_dispatched.wait(timeout=5)
            order.append("press")

        def on_release():
            order.append("release")

        listener = _build_nsevent_listener(on_press, on_release)
        listener.start()
        try:
            # Deliver a press (blocks until the release event is seen).
            deliver_flags_nsevent(listener, CONTROL | OPTION)
            # is_pressed is set synchronously in the event handler, so the
            # release event is now recognised.
            assert listener.is_pressed is True
            deliver_flags_nsevent(listener, 0)
            release_dispatched.set()
            listener._drain_for_tests()
        finally:
            listener.stop()

        assert order == ["press", "release"]

    def test_event_handler_does_not_run_callbacks_inline(self, capsys):
        """Delivering an event must return before the callback runs: the
        handler enqueues and never executes callbacks inline."""
        called = threading.Event()

        def on_press():
            called.set()

        listener = _build_nsevent_listener(on_press, MagicMock())
        listener.start()
        try:
            deliver_flags_nsevent(listener, CONTROL | OPTION)
            # The event handler has returned; if callbacks ran inline this
            # would already be set.
            assert not called.is_set()
            listener._drain_for_tests()
        finally:
            listener.stop()

        assert called.is_set()
