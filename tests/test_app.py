"""Hardware-free unit tests for the CLI app's hotkey release path (issue #17).

A release with no active recording must be a clean no-op: it must not
call the recorder, must not spawn a transcription thread, and must not
mutate recording state.
"""

import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the pyobjc / mlx / pyaudio / parakeet stack before importing
    kuiskaus.app so the module import never touches hardware.

    AppKit and PyObjCTools need real attributes (not a bare empty
    module), because kuiskaus/__init__.py unconditionally imports
    kuiskaus.hotkey_listener (``from AppKit import NSEvent`` /
    ``from PyObjCTools import AppHelper``) and kuiskaus.text_inserter
    (``from AppKit import NSPasteboard, NSPasteboardTypeString``) at
    module scope. When this file runs standalone -- no other test module
    has already imported the real kuiskaus package first, which is what
    quietly papers over this when the whole suite runs together -- a
    bare empty AppKit stub makes that import raise ImportError before
    kuiskaus.app is ever reached. None of these symbols are exercised at
    runtime here: HotkeyListener/AudioRecorder/TextInserter are all
    replaced with mocks in _make_app(), so a placeholder is sufficient.
    """
    appkit = types.ModuleType("AppKit")
    appkit.NSEvent = MagicMock(name="NSEvent")
    appkit.NSPasteboard = MagicMock(name="NSPasteboard")
    appkit.NSPasteboardTypeString = MagicMock(name="NSPasteboardTypeString")
    pyobjc_tools = types.ModuleType("PyObjCTools")
    pyobjc_tools.AppHelper = MagicMock(name="AppHelper")

    for name in (
        "pyaudio",
        "mlx_whisper",
        "mlx_whisper.load_models",
        "mlx.core",
        "ApplicationServices",
        "Quartz",
        "Foundation",
        "parakeet_mlx",
        "parakeet_mlx.audio",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "PyObjCTools", pyobjc_tools)


def _make_app():
    """Build a KuiskausApp with every component mocked out."""
    import kuiskaus.app as app_module
    from kuiskaus.app import KuiskausApp

    class MockTranscriber:
        def transcribe(self, audio, **kwargs) -> dict:
            return {"text": ""}

        def cleanup(self) -> None:
            pass

    mock_transcriber = MockTranscriber()
    with (
        patch.object(app_module, "AudioRecorder"),
        patch.object(app_module, "ParakeetTranscriber", return_value=mock_transcriber),
        patch.object(app_module, "TextInserter"),
        patch.object(app_module, "HotkeyListener"),
    ):
        return KuiskausApp()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    _install_stubs(monkeypatch)
    return _make_app()


class TestHotkeyReleaseNoOp:
    """A release with no active recording must be a clean no-op (#17)."""

    def test_release_without_press_is_noop(self, app, capsys):
        """Release with is_recording False: no recorder calls, Ready printed."""
        app.audio_recorder.start_recording = MagicMock()
        app.audio_recorder.stop_recording = MagicMock(return_value=None)

        app.on_hotkey_release()

        app.audio_recorder.start_recording.assert_not_called()
        app.audio_recorder.stop_recording.assert_not_called()
        assert app.is_recording is False
        assert app.recording_start_time is None
        out = capsys.readouterr().out
        assert "ready" in out.lower()
        assert "Stopped recording" not in out

    def test_release_without_press_starts_nothing(self, app):
        """A stray release must not flip any recording state."""
        app.on_hotkey_release()
        assert app.is_recording is False
        assert app.recording_start_time is None

    def test_release_after_press_still_stops(self, app):
        """The normal path still works: press then release stops recording."""
        app.audio_recorder.stop_recording = MagicMock(return_value=b"")

        app.on_hotkey_press()
        assert app.is_recording is True
        assert app.recording_start_time is not None

        app.on_hotkey_release()

        assert app.is_recording is False
        app.audio_recorder.start_recording.assert_called_once()
        app.audio_recorder.stop_recording.assert_called_once()

    def test_release_clears_start_time_on_noop(self, app):
        """The no-op path must also reset recording_start_time."""
        app.recording_start_time = time.time()  # stale value

        app.on_hotkey_release()

        assert app.is_recording is False
        assert app.recording_start_time is None
