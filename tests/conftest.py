"""Pytest collection config for the unit test suite.

The three manual hardware test scripts below require a microphone, loaded
models, or real system permissions. They are not unit tests: they are run
manually via `./run_tests.sh --hardware`.

Use collect_ignore (not addopts --ignore) so collection stays independent
of pyaudio availability — tests/test_audio.py imports pyaudio at module top.
"""

from pathlib import Path

import pytest

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
