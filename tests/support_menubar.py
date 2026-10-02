"""Shared helpers for the test_menubar test modules (issue #70).

Hardware-free unit tests for KuiskausMenuBarApp hotkey callbacks.

The listener classes and heavy components are mocked via monkeypatch so
the menubar module import never touches Quartz, pyaudio, or model
loading.
"""

import sys
import threading
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


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
        import kuiskaus.model_reload as _reload

        monkeypatch.setattr(_reload, "WhisperTranscriber", whisper_cls)
        monkeypatch.setattr(_reload, "ParakeetTranscriber", parakeet_cls)
    except ImportError:
        pass  # menubar not imported yet; the sys.modules stub covers it

    fake_text = _FakeTextInserterModule("kuiskaus.text_inserter")
    fake_text.TextInserter = MagicMock()

    monkeypatch.setitem(sys.modules, "kuiskaus.audio_recorder", fake_audio)
    monkeypatch.setitem(sys.modules, "kuiskaus.parakeet_transcriber", fake_parakeet)
    monkeypatch.setitem(sys.modules, "kuiskaus.whisper_transcriber", fake_whisper)
    monkeypatch.setitem(sys.modules, "kuiskaus.text_inserter", fake_text)


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
