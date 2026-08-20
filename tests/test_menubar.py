"""Hardware-free unit tests for KuiskausMenuBarApp hotkey callbacks.

The listener classes and heavy components are mocked via monkeypatch so
the menubar module import never touches Quartz, pyaudio, or model
loading.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import rumps

import kuiskaus.hotkey_listener_cgevent as hlcgevent


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub hardware/model dependencies before importing menubar."""
    fake_quartz = types.ModuleType("Quartz")
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    fake_audio = types.ModuleType("kuiskaus.audio_recorder")
    fake_audio.AudioRecorder = MagicMock()

    fake_parakeet = types.ModuleType("kuiskaus.parakeet_transcriber")
    fake_parakeet.ParakeetTranscriber = MagicMock()

    fake_whisper = types.ModuleType("kuiskaus.whisper_transcriber")
    fake_whisper.WhisperTranscriber = MagicMock()

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
    instance.total_transcriptions = 0
    instance.total_recording_time = 0.0
    instance.audio_recorder = MagicMock()
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
