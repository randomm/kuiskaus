"""Unit tests for the single-threaded callback executor in HotkeyListener (issue #17).

Hardware-free: the macOS frameworks (Quartz / AppKit / PyObjCTools) are
stubbed before import, and keyboard events are simulated via fakes fed to
``_handle_event`` directly.
"""

import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock

# --- stub macOS-only frameworks before importing the module under test ---

appkit = types.ModuleType("AppKit")
appkit.NSEvent = MagicMock(name="NSEvent")
appkit.NSApplication = MagicMock(name="NSApplication")
appkit.NSApp = MagicMock(name="NSApp")
appkit.NSPasteboard = MagicMock(name="NSPasteboard")
appkit.NSPasteboardTypeString = MagicMock(name="NSPasteboardTypeString")
pyobjc_tools = types.ModuleType("PyObjCTools")
pyobjc_tools.AppHelper = MagicMock(name="AppHelper")

sys.modules.setdefault("Quartz", types.ModuleType("Quartz"))
sys.modules.setdefault("AppKit", appkit)
sys.modules.setdefault("PyObjCTools", pyobjc_tools)

NS_FLAGS_CHANGED_MASK = 1 << 12
CONTROL = 1 << 18
OPTION = 1 << 19
COMMAND = 1 << 20


def make_fake_event(pressed: bool):
    """Fake NSEvent for a ⌃⌥ modifier change.

    pressed=True  -> Control+Option held (NSFlagsChanged with both masks)
    pressed=False -> modifiers released
    """
    event = MagicMock()
    event.type.return_value = NS_FLAGS_CHANGED_MASK
    event.modifierFlags.return_value = (CONTROL | OPTION) if pressed else 0
    return event


class TestHotkeyListenerExecutor(unittest.TestCase):
    """Press/release callbacks must run in event order on one worker thread."""

    def _make_listener(self, events):
        from kuiskaus.hotkey_listener import HotkeyListener

        return HotkeyListener(
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )

    def _feed(self, listener, *pressed_flags):
        for pressed in pressed_flags:
            listener._handle_event(make_fake_event(pressed))
        listener.stop_worker()

    def test_release_after_press_runs_in_order(self):
        events = []
        listener = self._make_listener(events)
        self._feed(listener, True, False)
        self.assertEqual(events, ["press", "release"])

    def test_release_before_press_is_noop(self):
        """A release with no prior press must not fire any callback."""
        events = []
        listener = self._make_listener(events)
        self._feed(listener, False, True, False)
        self.assertEqual(events, ["press", "release"])

    def test_press_press_release_runs_once_each(self):
        """A second press while already held must not queue a second press."""
        events = []
        listener = self._make_listener(events)
        self._feed(listener, True, True, False)
        self.assertEqual(events, ["press", "release"])

    def test_callbacks_run_on_worker_thread(self):
        """Callbacks run on the shared worker, not the main (event) thread."""
        from kuiskaus.hotkey_listener import HotkeyListener

        worker_threads = []
        listener = HotkeyListener(
            on_press=lambda: worker_threads.append(threading.current_thread()),
            on_release=lambda: None,
        )
        worker_name = listener._worker.name
        listener._handle_event(make_fake_event(True))
        listener.stop_worker()
        self.assertEqual(len(worker_threads), 1)
        self.assertIsNot(worker_threads[0], threading.main_thread())
        self.assertEqual(worker_threads[0].name, worker_name)

    def test_release_waits_for_blocking_press(self):
        """A release queued while the press blocks cannot run first."""
        from kuiskaus.hotkey_listener import HotkeyListener

        events = []
        press_done = threading.Event()

        def blocking_press():
            events.append("press")
            press_done.wait(timeout=2.0)

        listener = HotkeyListener(
            on_press=blocking_press, on_release=lambda: events.append("release")
        )
        listener._handle_event(make_fake_event(True))
        # Wait until the press callback is actually running, then queue the
        # release while it is still blocked on the worker thread.
        time.sleep(0.05)
        listener._handle_event(make_fake_event(False))
        press_done.set()
        listener.stop_worker()
        # Single-threaded executor: release always runs after the press.
        self.assertEqual(events, ["press", "release"])

    def test_stop_worker_flushes_queue(self):
        """stop_worker runs pending callbacks before terminating the worker."""
        from kuiskaus.hotkey_listener import HotkeyListener

        events = []
        listener = HotkeyListener(
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        listener._handle_event(make_fake_event(True))
        listener._handle_event(make_fake_event(False))
        # No sleep — stop_worker must drain both before the thread ends.
        listener.stop_worker()
        self.assertEqual(events, ["press", "release"])
        self.assertIsNone(listener._worker)

    def test_executor_is_queue_plus_worker_thread(self):
        """The executor is a queue + dedicated worker thread (issue #17 DoD)."""
        from kuiskaus.hotkey_listener import HotkeyListener

        listener = HotkeyListener(on_press=lambda: None, on_release=lambda: None)
        self.assertIsInstance(listener._queue, queue.Queue)
        self.assertIsInstance(listener._worker, threading.Thread)
        self.assertTrue(listener._worker.daemon)
        listener.stop_worker()


if __name__ == "__main__":
    unittest.main()
