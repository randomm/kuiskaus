"""Unit tests for HotkeyListener dispatch order (issue #17).

Hardware-free: the ``kuiskaus`` package and the pyobjc / pyaudio / mlx
dependencies are stubbed in sys.modules via monkeypatch around the import
of the module under test, so the real native modules are never loaded.
Keyboard events are simulated by feeding fakes to ``_handle_event``
directly.

``importlib`` is used to load the module file directly (bypassing the
package ``__init__.py``) so the heavy transcriber / audio / text
submodules are never imported.
"""

import importlib.util
import os
import sys
import threading
from types import ModuleType
from unittest.mock import MagicMock

import pytest

NS_FLAGS_CHANGED_MASK = 1 << 12
CONTROL = 1 << 18
OPTION = 1 << 19

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "kuiskaus", "hotkey_listener.py"
)


def _make_appkit_stub() -> ModuleType:
    appkit = ModuleType("AppKit")

    class _FakeNSEvent:
        @staticmethod
        def addLocalMonitorForEventsMatchingMask_handler_(*a, **kw):
            raise RuntimeError("monitoring disabled in tests")

        @staticmethod
        def addGlobalMonitorForEventsMatchingMask_handler_(*a, **kw):
            raise RuntimeError("monitoring disabled in tests")

        @staticmethod
        def removeMonitor_(*a, **kw):
            raise RuntimeError("monitoring disabled in tests")

    appkit.NSEvent = _FakeNSEvent
    return appkit


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Stub the kuiskaus package chain and load hotkey_listener directly."""
    # Load the real callback_dispatcher (stdlib-only) so the module under
    # test can import it normally.
    dispatcher_spec = importlib.util.spec_from_file_location(
        "kuiskaus.callback_dispatcher",
        os.path.join(
            os.path.dirname(__file__), "..", "kuiskaus", "callback_dispatcher.py"
        ),
    )
    dispatcher_mod = importlib.util.module_from_spec(dispatcher_spec)
    monkeypatch.setitem(sys.modules, "kuiskaus.callback_dispatcher", dispatcher_mod)
    dispatcher_spec.loader.exec_module(dispatcher_mod)

    # Stub the heavy submodules.
    for name in (
        "kuiskaus",
        "kuiskaus.audio_recorder",
        "kuiskaus.hotkey_listener",
        "kuiskaus.parakeet_transcriber",
        "kuiskaus.text_inserter",
        "kuiskaus.transcriber",
        "kuiskaus.voxtral_transcriber",
        "kuiskaus.whisper_transcriber",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    monkeypatch.setitem(sys.modules, "AppKit", _make_appkit_stub())
    pyobjc = ModuleType("PyObjCTools")
    pyobjc.AppHelper = MagicMock(name="AppHelper")
    monkeypatch.setitem(sys.modules, "PyObjCTools", pyobjc)

    spec = importlib.util.spec_from_file_location(
        "kuiskaus.hotkey_listener", _MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "kuiskaus.hotkey_listener", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def listener_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _install_stubs(monkeypatch)


def make_fake_event(pressed: bool) -> MagicMock:
    """Fake NSEvent for a ⌃⌥ modifier change."""
    event = MagicMock()
    event.type.return_value = NS_FLAGS_CHANGED_MASK
    event.modifierFlags.return_value = (CONTROL | OPTION) if pressed else 0
    return event


def _make_listener(listener_module, on_press=None, on_release=None):
    return listener_module.HotkeyListener(
        on_press=on_press or (lambda: None),
        on_release=on_release or (lambda: None),
    )


def _feed(listener, *pressed_flags):
    listener._dispatcher.start()
    for pressed in pressed_flags:
        listener._handle_event(make_fake_event(pressed))
    listener._dispatcher.stop()


class TestHotkeyListenerDispatch:
    """Press/release callbacks must run in event order on one worker."""

    def test_release_after_press_runs_in_order(self, listener_module):
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        _feed(listener, True, False)
        assert events == ["press", "release"]

    def test_release_before_press_is_noop(self, listener_module):
        """A release with no prior press must not fire any callback."""
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        _feed(listener, False, True, False)
        assert events == ["press", "release"]

    def test_press_press_release_runs_once_each(self, listener_module):
        """A second press while already held must not queue a second press."""
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        _feed(listener, True, True, False)
        assert events == ["press", "release"]

    def test_callbacks_run_on_worker_thread(self, listener_module):
        """Callbacks run on the dispatcher worker, not the main thread."""
        worker_threads = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: worker_threads.append(threading.current_thread()),
        )
        listener._dispatcher.start()
        listener._handle_event(make_fake_event(True))
        listener._dispatcher.stop()
        assert len(worker_threads) == 1
        assert worker_threads[0] is not threading.main_thread()

    def test_release_waits_for_blocking_press(self, listener_module):
        """A release queued while the press blocks cannot run first."""
        events: list[str] = []
        press_started = threading.Event()
        press_done = threading.Event()

        def blocking_press():
            events.append("press")
            press_started.set()
            press_done.wait(timeout=2.0)

        listener = _make_listener(
            listener_module,
            on_press=blocking_press,
            on_release=lambda: events.append("release"),
        )
        listener._dispatcher.start()
        listener._handle_event(make_fake_event(True))
        # Wait until the press callback is actually running, then queue the
        # release while it is still blocked on the worker thread.
        assert press_started.wait(timeout=2.0)
        listener._handle_event(make_fake_event(False))
        press_done.set()
        listener._dispatcher.stop()
        # Single-threaded executor: release always runs after the press.
        assert events == ["press", "release"]

    def test_stop_flushes_queue(self, listener_module):
        """stop() runs pending callbacks before terminating the worker."""
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        listener._dispatcher.start()
        listener._handle_event(make_fake_event(True))
        listener._handle_event(make_fake_event(False))
        # No sleep — stop() must drain both before the thread ends.
        listener._dispatcher.stop()
        assert events == ["press", "release"]

    def test_stop_is_idempotent(self, listener_module):
        """stop() must be safe to call twice."""
        listener = _make_listener(listener_module)
        listener._dispatcher.start()
        listener._dispatcher.stop()
        listener._dispatcher.stop()  # must not raise

    def test_run_loop_guard_survives(self, listener_module):
        """A raising event handler must not crash the run loop."""
        listener = _make_listener(listener_module)
        listener._dispatcher.start()
        # A broken event must be swallowed by the run-loop guard.
        listener._handle_event(MagicMock())
        # A real (but unhandled) event type must not raise either.
        listener._handle_event(make_fake_event(False))
        listener._dispatcher.stop()
