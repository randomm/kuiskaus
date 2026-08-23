"""Unit tests for HotkeyListenerCGEvent callback ordering (issue #17).

The CGEvent tap callback must never run user callbacks inline (it would
block the run loop) and must never race press/release on independent
threads. Both callbacks are therefore enqueued onto a single-threaded
worker (the shared :class:`CallbackDispatcher`) so a release always
observes the state left by its press.
"""

import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

CONTROL_OPTION_FLAGS = (1 << 18) | (1 << 19)
CONTROL_OPTION_COMMAND_FLAGS = (1 << 18) | (1 << 19) | (1 << 20)
RELEASE_FLAGS = 0

# Quartz.kCGEventFlagsChanged — must match the real constant, the tap
# callback gates on it.
FLAGS_CHANGED_TYPE = 12

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "kuiskaus", "hotkey_listener_cgevent.py"
)


class _FakeQuartz(ModuleType):
    def __init__(self) -> None:
        super().__init__("Quartz")
        self.kCGEventFlagsChanged = FLAGS_CHANGED_TYPE
        self.CGEventGetFlags = MagicMock()


def _make_package_stub() -> ModuleType:
    """Minimal package stand-in for the `kuiskaus` name (relative imports
    resolve against its __path__)."""
    pkg = ModuleType("kuiskaus")
    pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "kuiskaus")]
    return pkg


def _load_real_submodule(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    """Load a stdlib-only kuiskaus submodule from disk into sys.modules."""
    rel = os.path.join("..", *name.split("."))
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(os.path.dirname(__file__), rel + ".py"),
        submodule_search_locations=[os.path.join(os.path.dirname(__file__), rel)],
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Stub the kuiskaus package chain and load the module directly."""
    # Load the real stdlib-only submodules (callback_dispatcher, debug) so
    # the module under test can import them normally.
    for name in ("kuiskaus.callback_dispatcher", "kuiskaus.debug"):
        _load_real_submodule(monkeypatch, name)

    # Stub the heavy submodules (and the package itself, as a minimal
    # package stub so relative imports resolve against its __path__).
    for name in (
        "kuiskaus.audio_recorder",
        "kuiskaus.hotkey_listener",
        "kuiskaus.hotkey_listener_cgevent",
        "kuiskaus.parakeet_transcriber",
        "kuiskaus.text_inserter",
        "kuiskaus.transcriber",
        "kuiskaus.voxtral_transcriber",
        "kuiskaus.whisper_transcriber",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "kuiskaus", _make_package_stub())

    fake_quartz = _FakeQuartz()
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    spec = importlib.util.spec_from_file_location(
        "kuiskaus.hotkey_listener_cgevent", _MODULE_PATH
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "kuiskaus.hotkey_listener_cgevent", mod)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def listener_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _install_stubs(monkeypatch)


def _make_listener(listener_module, on_press=None, on_release=None):
    return listener_module.HotkeyListenerCGEvent(
        on_press=on_press or (lambda: None),
        on_release=on_release or (lambda: None),
    )


def _press_event(listener):
    with patch("Quartz.CGEventGetFlags", return_value=CONTROL_OPTION_FLAGS):
        listener._event_tap_callback(None, FLAGS_CHANGED_TYPE, MagicMock(), None)


def _release_event(listener):
    with patch("Quartz.CGEventGetFlags", return_value=RELEASE_FLAGS):
        listener._event_tap_callback(None, FLAGS_CHANGED_TYPE, MagicMock(), None)


class TestEventTapDispatch:
    def test_tap_passthrough(self, listener_module):
        listener = _make_listener(listener_module)
        with patch("Quartz.CGEventGetFlags", return_value=RELEASE_FLAGS):
            result = listener._event_tap_callback(None, 0, MagicMock(), None)
        assert result is not None
        listener._dispatcher.stop()

    def test_press_enqueues_release_enqueues(self, listener_module):
        listener = _make_listener(listener_module)
        _press_event(listener)
        assert listener.is_pressed is True
        assert listener._dispatcher._queue.qsize() == 1
        _release_event(listener)
        assert listener.is_pressed is False
        assert listener._dispatcher._queue.qsize() == 2
        listener._dispatcher.stop()

    def test_command_suppresses_press(self, listener_module):
        listener = _make_listener(listener_module)
        with patch(
            "Quartz.CGEventGetFlags",
            return_value=CONTROL_OPTION_COMMAND_FLAGS,
        ):
            listener._event_tap_callback(None, 0, MagicMock(), None)
        assert listener.is_pressed is False
        assert listener._dispatcher._queue.empty()
        listener._dispatcher.stop()

    def test_callback_raising_keeps_run_loop_alive(self, listener_module):
        """A raising press callback must not kill the dispatcher worker.

        The worker thread is started once per listener lifetime, not per
        event, so if a callback exception ever kills it, every later
        hotkey event is silently dropped until the app restarts. Both
        the worker's liveness and its ability to keep processing
        subsequent callbacks must be proven, not just that enqueuing
        doesn't raise.
        """
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=MagicMock(side_effect=RuntimeError("boom")),
            on_release=lambda: events.append("release"),
        )
        listener._dispatcher.start()
        try:
            _press_event(listener)
            listener._dispatcher.drain()
            assert listener._dispatcher._worker.is_alive()

            # Prove the loop actually survived: a later callback must
            # still run, not just that the thread object exists.
            _release_event(listener)
            listener._dispatcher.drain()
            assert events == ["release"]
        finally:
            listener._dispatcher.stop()

    def test_non_flags_event_ignored(self, listener_module):
        listener = _make_listener(listener_module)
        listener._event_tap_callback(None, "some-other-type", MagicMock(), None)
        assert listener._dispatcher._queue.empty()
        listener._dispatcher.stop()


class TestDispatcherOrdering:
    """The shared single-threaded dispatcher must run callbacks in order."""

    def test_release_before_press_state_is_noop(self, listener_module):
        """A release with no prior press enqueues nothing."""
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        listener._dispatcher.start()
        try:
            _release_event(listener)  # nothing pressed: ignored
            _press_event(listener)
            _release_event(listener)
            listener._dispatcher.drain()
        finally:
            listener._dispatcher.stop()
        assert events == ["press", "release"]

    def test_callbacks_run_in_event_order(self, listener_module):
        """Many interleaved events dispatch strictly in enqueue order."""
        order: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: order.append("press"),
            on_release=lambda: order.append("release"),
        )
        listener._dispatcher.start()
        try:
            for _ in range(25):
                _press_event(listener)
                _release_event(listener)
            listener._dispatcher.drain()
        finally:
            listener._dispatcher.stop()
        assert order == ["press", "release"] * 25

    def test_release_observess_press_state(self, listener_module):
        """The classic #17 race: release must see the press's state."""
        events: list[str] = []
        recording = False
        seen = {}

        def on_press():
            nonlocal recording
            recording = True
            events.append("press")

        def on_release():
            nonlocal recording
            seen["was_recording"] = recording
            events.append("release")
            if recording:
                recording = False

        listener = _make_listener(
            listener_module, on_press=on_press, on_release=on_release
        )
        listener._dispatcher.start()
        try:
            _press_event(listener)
            _release_event(listener)
            listener._dispatcher.drain()
        finally:
            listener._dispatcher.stop()
        assert events == ["press", "release"]
        assert seen["was_recording"] is True

    def test_press_press_release_only_first_press_fires(self, listener_module):
        events: list[str] = []
        listener = _make_listener(
            listener_module,
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        listener._dispatcher.start()
        try:
            _press_event(listener)
            _press_event(listener)  # ignored: already pressed
            _release_event(listener)
            listener._dispatcher.drain()
        finally:
            listener._dispatcher.stop()
        assert events == ["press", "release"]

    def test_dispatcher_stops_cleanly(self, listener_module):
        listener = _make_listener(listener_module)
        listener._dispatcher.start()
        listener._dispatcher.stop()
        # No hang, no exceptions: success.


class TestDebugGating:
    """Per-event [DEBUG] prints are opt-in via KUISKAUS_DEBUG (issue #22).

    Both tests drive the same event pair and assert the same two strings
    (presence vs. exact absence), so a regression in the gating
    expression fails in either direction.
    """

    PRESS_PRINT = "[DEBUG CGEvent] Hotkey pressed! (CGEvent)"
    RELEASE_PRINT = "[DEBUG CGEvent] Hotkey released! (CGEvent)"

    def test_debug_output_off_by_default(self, listener_module, capsys):
        listener = _make_listener(listener_module)
        _press_event(listener)
        _release_event(listener)
        captured = capsys.readouterr().out
        assert self.PRESS_PRINT not in captured
        assert self.RELEASE_PRINT not in captured

    def test_debug_output_on_when_enabled(self, listener_module, monkeypatch, capsys):
        monkeypatch.setattr(listener_module, "DEBUG", True)
        listener = _make_listener(listener_module)
        _press_event(listener)
        _release_event(listener)
        captured = capsys.readouterr().out
        assert self.PRESS_PRINT in captured
        assert self.RELEASE_PRINT in captured
