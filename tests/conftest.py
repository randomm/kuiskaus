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
