"""Pytest collection config for the unit test suite.

The three manual hardware test scripts below require a microphone, loaded
models, or real system permissions. They are not unit tests: they are run
manually via `./run_tests.sh --hardware`.

Use collect_ignore (not addopts --ignore) so collection stays independent
of pyaudio availability — tests/test_audio.py imports pyaudio at module top.

Collect_ignore entries are validated at configure time: the check raises
if an entry no longer matches a file on disk. An entry must therefore land
in the same PR as the file it names (deleting a hardware script and
updating this list atomically), so a stale entry can never silently let a
hardware script into collection.
"""

import importlib
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest
import rumps

from kuiskaus.transcriber import Transcriber
from tests import support_menubar, support_text_inserter
from tests.support_audio_recorder import (
    _cleanup_recorder,
)

collect_ignore: list[str] = [
    "test_audio.py",
    "test_whisper.py",
    "test_integration.py",
]

_TESTS_DIR = Path(__file__).parent


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast if a collect_ignore entry no longer matches a file on disk.

    A stale entry (script moved to a subdirectory) would let pytest
    collect a manual hardware script that imports hardware deps at module
    top and abort collection wherever the dep is missing (the failure
    #20 fixed). Entries that still point at a real file are valid
    regardless of whether that file is a hardware script or a unit test.
    """
    stale = [name for name in collect_ignore if not (_TESTS_DIR / name).is_file()]
    if stale:
        raise RuntimeError(
            f"conftest.collect_ignore entries no longer match a file on disk: {stale}"
        )


@pytest.fixture(autouse=True)
def _clean_kuiskaus_debug_env(monkeypatch: pytest.MonkeyPatch):
    """Reset KUISKAUS_DEBUG for every test.

    The hotkey listeners read it once at import (issue #22); tests that
    set the env var (e.g. the env-var wiring test) must not leak it into
    other tests' module loads.
    """
    monkeypatch.delenv("KUISKAUS_DEBUG", raising=False)


@pytest.fixture(autouse=True)
def coreaudio_stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the PyObjC CoreAudio bridge for every test (issue #72).

    autouse, not per-test monkeypatch.setitem: kuiskaus.audio_device
    resolves CoreAudio through sys.modules at call time, and a stub that
    only some tests install would let the host machine's real default
    input device leak into the suite (a live read of the developer's
    microphone is never a legitimate test input).

    Default behaviour: every property read fails, so
    read_live_default_input() returns None and the recorder takes the
    no-reset fast path -- the behaviour every pre-#72 recorder test was
    written against. Tests that want a live default call
    tests.support_audio_recorder._live_default_input(stub, device_id).
    """
    stub = MagicMock(name="CoreAudio")
    from kuiskaus import audio_coreaudio

    # The selector constants are the contract between this stub and the
    # reader; restating the literals here meant a test could pass against a
    # number the module no longer uses.
    stub.kAudioObjectSystemObject = audio_coreaudio.kAudioObjectSystemObject
    stub.kAudioObjectPropertyScopeGlobal = (
        audio_coreaudio.kAudioObjectPropertyScopeGlobal
    )
    stub.kAudioObjectPropertyElementMain = (
        audio_coreaudio.kAudioObjectPropertyElementMain
    )
    stub.kAudioHardwarePropertyDefaultInputDevice = (
        audio_coreaudio.kAudioHardwarePropertyDefaultInputDevice
    )
    stub.kAudioDevicePropertyDeviceIsAlive = (
        audio_coreaudio.kAudioDevicePropertyDeviceIsAlive
    )
    # A real AudioObjectPropertyAddress carries the selector the reader
    # asked for, so a stub that answers per-selector can tell the two
    # reads apart -- which is what makes "device moved" and "device not
    # alive" separately testable.
    stub.AudioObjectPropertyAddress.side_effect = lambda selector, scope, element: (
        SimpleNamespace(mSelector=selector, mScope=scope, mElement=element)
    )
    stub.AudioObjectGetPropertyData.return_value = (1852797029, 0, None)
    monkeypatch.setitem(sys.modules, "CoreAudio", stub)
    # The unavailable-framework latch and the OSStatus log budget are
    # process-lifetime state; a test that tripped either must not hand the
    # verdict to the rest of the suite.
    audio_coreaudio._reset_diagnostics_for_tests()
    return stub


@pytest.fixture
def audio_recorder_module(monkeypatch: pytest.MonkeyPatch):
    """Reload kuiskaus.audio_recorder bound to a stubbed pyaudio module.

    Self-undoing: reloaded again against the real pyaudio package on
    teardown, so the stub can never leak into tests outside this file.
    Recorders built during the test are cleaned up explicitly so their
    __del__-time cleanup() (unbounded join) can never wedge interpreter
    shutdown.
    """
    fake_pyaudio = MagicMock(name="pyaudio")
    fake_pyaudio.paInt16 = 8
    monkeypatch.setitem(sys.modules, "pyaudio", fake_pyaudio)

    import kuiskaus.audio_recorder as module

    importlib.reload(module)

    built: list = []
    orig_audio_recorder = module.AudioRecorder

    def tracker(*args, **kwargs):
        recorder = orig_audio_recorder(*args, **kwargs)
        built.append(recorder)
        return recorder

    patcher = mock.patch.object(module, "AudioRecorder", side_effect=tracker)
    patcher.start()

    yield module

    patcher.stop()

    # Deterministically tear down every worker before the recorder objects
    # are dropped, so the unbounded join in __del__ -> cleanup() can never
    # run at GC/interpreter-shutdown time (where it deadlocks while any
    # other thread is alive).
    for recorder in built:
        _cleanup_recorder(recorder)
    built.clear()
    monkeypatch.undo()
    importlib.reload(module)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    support_menubar._install_stubs(monkeypatch)
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


@pytest.fixture
def inserter(monkeypatch: pytest.MonkeyPatch):
    support_text_inserter._install_stubs(monkeypatch)
    # Re-import the module fresh so it binds THIS test's AppKit stub.
    # (sys.modules["kuiskaus.text_inserter"] may hold a prior test's
    # module object whose globals point at a different stub.)
    import kuiskaus.text_inserter as ti

    importlib.reload(ti)
    # Default: subprocess.run is a benign success so the osascript call
    # (issue #58: the sole insertion path) never shells out in the test
    # environment. monkeypatch restores the real subprocess.run after
    # each test; failure tests replace this mock per-test via
    # _patch_subprocess.
    monkeypatch.setattr(
        ti.subprocess,
        "run",
        MagicMock(
            return_value=MagicMock(returncode=0, stderr=""), name="subprocess.run"
        ),
    )
    return ti.TextInserter()


@pytest.fixture
def pasteboard(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The NSPasteboard stub the current text_inserter module is bound to."""
    import kuiskaus.text_inserter as ti

    pb = MagicMock(name="pasteboard")
    ti.NSPasteboard.generalPasteboard.return_value = pb
    return pb
