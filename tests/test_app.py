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


class _FakeAppKit(types.ModuleType):
    NSEvent: MagicMock
    NSPasteboard: MagicMock
    NSPasteboardTypeString: MagicMock


class _FakePyObjCTools(types.ModuleType):
    AppHelper: MagicMock


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
    appkit = _FakeAppKit("AppKit")
    appkit.NSEvent = MagicMock(name="NSEvent")
    appkit.NSPasteboard = MagicMock(name="NSPasteboard")
    appkit.NSPasteboardTypeString = MagicMock(name="NSPasteboardTypeString")
    pyobjc_tools = _FakePyObjCTools("PyObjCTools")
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
    instance = _make_app()
    # Real AudioRecorder.__init__ sets last_error = None; a bare
    # MagicMock() attribute is truthy by default, which would make every
    # test believe a mic error is persisted unless overridden here.
    instance.audio_recorder.last_error = None
    return instance


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


class TestHotkeyPressAdmission:
    """start_recording()'s bool return must be consumed (#16)."""

    def test_press_refused_does_not_flip_recording_state(self, app):
        """A refused start_recording() must not set is_recording True."""
        app.audio_recorder.start_recording = MagicMock(return_value=False)

        app.on_hotkey_press()

        assert app.is_recording is False
        assert app.recording_start_time is None

    def test_press_admitted_flips_recording_state(self, app):
        """An admitted start_recording() still sets is_recording True."""
        app.audio_recorder.start_recording = MagicMock(return_value=True)

        app.on_hotkey_press()

        assert app.is_recording is True
        assert app.recording_start_time is not None


class TestTranscribeAndInsertNotification:
    """Insert-failure notification + success-notification suppression (#41)."""

    def _drive_transcribe(self, app, text: str):
        """Run the transcription worker synchronously with a known transcript."""
        import numpy as np

        class FixedTranscriber:
            def transcribe(self, audio, **kwargs) -> dict:
                return {"text": text}

            def cleanup(self) -> None:
                pass

        app.transcriber = FixedTranscriber()
        app.show_notification = MagicMock()
        app._transcribe_and_insert(np.array([0.1]), 1.0)

    def test_cli_notifies_on_insert_failure(self, app):
        """insert_text False + last_error set: 'Insert failed' notification,
        and the 'Transcribed' success notification must NOT fire."""
        app.text_inserter.insert_text = MagicMock(return_value=False)
        app.text_inserter.last_error = "CGEventPost key down failed"

        self._drive_transcribe(app, "hello world")

        titles = [c.args[0] for c in app.show_notification.call_args_list]
        assert titles == ["Transcribing", "Insert failed"], (
            f"unexpected notifications: {titles}"
        )
        assert "CGEventPost key down failed" in app.show_notification.call_args.args[1]

    def test_cli_success_notification_only_on_insert_success(self, app):
        """insert_text True: the 'Transcribed' notification fires as before."""
        app.text_inserter.insert_text = MagicMock(return_value=True)

        self._drive_transcribe(app, "hello world")

        titles = [c.args[0] for c in app.show_notification.call_args_list]
        assert titles == ["Transcribing", "Transcribed"], (
            f"unexpected notifications: {titles}"
        )


class TestHotkeyReleaseLastError:
    """A failed/stuck microphone open must surface via last_error (#16)."""

    def test_release_with_last_error_skips_transcription_and_notifies(self, app):
        """last_error set: no transcription thread, error notification shown."""
        app.audio_recorder.stop_recording = MagicMock(return_value=b"\x00\x00")
        app.audio_recorder.last_error = "Microphone unavailable: boom"
        app.show_notification = MagicMock()

        app.on_hotkey_press()
        app.show_notification.reset_mock()
        with patch("threading.Thread") as mock_thread_cls:
            app.on_hotkey_release()
            mock_thread_cls.assert_not_called()

        app.show_notification.assert_called_once()
        args = app.show_notification.call_args.args
        assert args[0] == "Error"
        assert "boom" in args[1]

    def test_release_without_last_error_still_transcribes(self, app):
        """No last_error: the normal transcription path still fires."""
        import numpy as np

        app.audio_recorder.stop_recording = MagicMock(
            return_value=np.array([0.1, 0.2], dtype=np.float32)
        )
        app.audio_recorder.last_error = None

        app.on_hotkey_press()
        with patch("threading.Thread") as mock_thread_cls:
            app.on_hotkey_release()
            mock_thread_cls.assert_called_once()
