"""Pytest collection config for the unit test suite.

The three manual hardware test scripts below require a microphone, loaded
models, or real system permissions. They are not unit tests: they are run
manually via `./run_tests.sh --hardware`.

Use collect_ignore (not addopts --ignore) so collection stays independent
of pyaudio availability — tests/test_audio.py imports pyaudio at module top.
"""

collect_ignore = ["test_audio.py", "test_whisper.py", "test_integration.py"]
