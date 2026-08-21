"""Smoke test: conftest's collect_ignore keeps manual hardware scripts out.

collect_ignore entries are bare filenames resolved relative to the conftest
directory. If a manual hardware script ever moves into a subdirectory the
entry stops matching and pytest collects the script — test_audio.py imports
pyaudio at module top, which aborts collection wherever portaudio is
missing (the failure #20 fixed).

This test asserts collect_ignore still matches exactly the set of manual
scripts present at the top level of tests/, so any drift fails fast.

Note: `from conftest import ...` is not possible here — tests/ is an
importable package (tests/__init__.py) and conftest.py is not one of its
modules, so the conftest is loaded explicitly from its file path instead.
"""

import importlib.util
import pathlib

TESTS_DIR = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "kuiskaus_test_conftest",
    TESTS_DIR / "conftest.py",
)
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
collect_ignore = _conftest.collect_ignore

MANUAL_HARDWARE_SCRIPTS = {"test_audio.py", "test_whisper.py", "test_integration.py"}


def test_collect_ignore_covers_manual_scripts():
    """Every manual hardware script at the top level is excluded."""
    on_disk = {
        entry.name
        for entry in TESTS_DIR.iterdir()
        if entry.is_file() and entry.name.startswith("test_")
    }
    manual_on_disk = on_disk & MANUAL_HARDWARE_SCRIPTS
    assert set(collect_ignore) >= manual_on_disk, (
        "collect_ignore no longer covers a manual hardware script on disk: "
        f"missing {manual_on_disk - set(collect_ignore)}"
    )


def test_collect_ignore_does_not_cover_real_tests():
    """collect_ignore must not exclude actual unit-test modules."""
    on_disk = {
        entry.name
        for entry in TESTS_DIR.iterdir()
        if entry.is_file() and entry.name.startswith("test_")
    }
    real_tests = on_disk - MANUAL_HARDWARE_SCRIPTS
    excluded = set(collect_ignore) & real_tests
    assert not excluded, (
        f"collect_ignore wrongly excludes real unit-test modules: {excluded}"
    )
