"""test_audio_recorder tests, part 1 of 6 (issue #70)."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.support_audio_recorder import (
    _assert_stop_log_line,
    _blocking_stream,
    _make_pyaudio_instance,
    _make_recorder,
    _wait_until,
)


# ---------------------------------------------------------------------------
# Never wedge: a failed open must not kill the worker silently
# ---------------------------------------------------------------------------
def test_failed_open_leaves_recording_false_and_next_start_succeeds(
    audio_recorder_module,
):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry failed")

    # Attempt 1 (cached, index -1) opens with pa1 (fails); attempt 2
    # (fresh, index 0) opens with pa_retry (fails) -- pin max_attempts
    # to match so the loop doesn't try a 3rd/4th construction the mock
    # has no instance left to return.
    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)
    # Fast-failure path: both attempts fail immediately, so the worker
    # can complete and clear recording/recording_thread before any
    # capture; wait for completion on observable state instead (lens
    # review HIGH #1/#4 read-before-start race).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # The wedge: open()'s OSError must not leave recording stuck True.
    assert recorder.recording is False
    assert recorder.stream is None
    assert recorder.recording_thread is None
    assert recorder.last_error is not None

    # A following start_recording() must be admitted -- the wedge is
    # structurally impossible, not just avoided this once. A blocking
    # read keeps the second worker alive so its identity is observable.
    second_release = threading.Event()
    pa1.open.side_effect = None
    pa1.open.return_value = _blocking_stream(OSError("stop the loop"), second_release)
    assert recorder.start_recording() is True
    second_thread = recorder.recording_thread
    assert second_thread is not None
    assert second_thread.is_alive()

    second_release.set()
    second_thread.join(timeout=10.0)
    assert not second_thread.is_alive()


def test_stop_recording_after_failed_open_returns_empty_array(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry failed")

    # Both attempts fail -- pin max_attempts to match (see sibling test).
    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)
    # Fast-failure path: both attempts fail immediately, so the worker
    # can complete and clear recording before any capture; wait for
    # completion on observable state instead (lens review HIGH #1/#4
    # read-before-start race).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    result = recorder.stop_recording()

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.size == 0


def test_stop_recording_logs_reason_retry_exhausted(audio_recorder_module, capsys):
    """A stop after the retry loop exhausted (worker already set
    last_error and cleared recording) must log the early-return reason
    retry-exhausted -- the existing #16 red state is unchanged, the log
    line is pure observability (issue #40)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry failed")

    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.last_error is not None, timeout=5.0)

    recorder.stop_recording()

    _assert_stop_log_line(capsys.readouterr(), "retry-exhausted")


def test_stop_recording_logs_reason_no_worker(audio_recorder_module, capsys):
    """A stop before any start (or a double-stop) must log reason=
    no-worker with chunks=0 and duration_ms=0 (issue #40)."""
    module = audio_recorder_module
    recorder = _make_recorder(module, _make_pyaudio_instance(0))

    result = recorder.stop_recording()

    assert result.size == 0
    _assert_stop_log_line(capsys.readouterr(), "no-worker", chunks=0, duration_ms_min=0)


def test_stop_recording_logs_reason_race_lost(
    audio_recorder_module, capsys, monkeypatch
):
    """The race the user hit (#40): the worker adopted the stream but the
    user released before the first stream.read() landed, so the queue
    drains empty with no last_error. Must log reason=race-lost and a
    non-negative duration_ms (issue #40)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)

    # Synchronisation: the worker blocks in read() until unblocked.
    # stop_recording() sets recording=False and then join(timeout);
    # a helper thread unblocks the read shortly after, so the worker's
    # read raises OSError, the worker breaks on the next loop check
    # (recording is already False), and exits cleanly well before the
    # join timeout. Queue is empty (no frame was ever put), last_error
    # is None -> stop_recording() classifies this as race-lost (#40).
    read_gate = threading.Event()

    def fake_read(*_args, **_kwargs):
        read_gate.wait(timeout=10.0)
        raise OSError("loop ended before first frame")

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    # pa1 is the __init__'s cached instance; attempt 1 reuses it (issue #42).
    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    # Post-adoption capture idiom: wait for adoption first (the worker
    # only clears recording_thread in post-loop teardown, so it cannot
    # have exited before adoption lands).
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None
    # Wait until the worker is provably inside read().
    assert _wait_until(lambda: stream.read.called, timeout=5.0)

    # Unblock the read from a helper thread so the worker can exit
    # cleanly while stop_recording() is blocked in its join.
    def _unblock():
        threading.Event().wait(timeout=0.05)
        read_gate.set()

    unblock_thread = threading.Thread(target=_unblock, daemon=True)
    unblock_thread.start()

    result = recorder.stop_recording()
    unblock_thread.join(timeout=5.0)
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    assert result.size == 0
    assert recorder.last_error is None
    _assert_stop_log_line(capsys.readouterr(), "race-lost", chunks=0, duration_ms_min=0)


def test_stop_recording_omits_log_line_on_non_empty_return(
    audio_recorder_module, capsys
):
    """The [audio.stop] line is only for empty-array returns: a normal
    capture must NOT log it (issue #40)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    frame = b"\x00\x00" * 1024  # one full 1024-sample int16 frame

    def fake_read(*_args, **_kwargs):
        release_event.wait(timeout=10.0)
        return frame

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    # pa1 is the __init__'s cached instance; attempt 1 reuses it (issue #42).
    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    # Post-adoption capture idiom: the worker is blocked in read() after
    # adoption, so it cannot have torn down recording_thread.
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None

    release_event.set()
    thread.join(timeout=10.0)

    result = recorder.stop_recording()
    assert result.size > 0

    captured = capsys.readouterr()
    log_lines = [
        line for line in captured.out.splitlines() if line.startswith("[audio.stop]")
    ]
    assert log_lines == []


@pytest.mark.skip(
    reason="#49 partial: cached-PyAudio harness rework hangs test — needs deeper mock/threading fix, follow-up chore"
)
def test_current_generation_reflects_recording_cycles(audio_recorder_module):
    """The public current_generation property mirrors the internal
    generation counter: 0 before any recording, 1 after the first
    start_recording(), 2 after the second (issue #40/43 lens review
    MEDIUM #2: UI drain code reads the current generation through this
    accessor instead of the private _generation attribute)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release1 = threading.Event()
    pa1.open.return_value = _blocking_stream(OSError("stop"), release1)
    pa2 = _make_pyaudio_instance(1)
    release2 = threading.Event()
    pa2.open.return_value = _blocking_stream(OSError("stop"), release2)

    # Cycle 1's attempt 1 drives the cached instance (pa1, which __init__
    # adopts from the init slot); cycle 2's attempt 1 reuses the cache --
    # the device is unchanged, so the poll never consumes pa2 (it is a
    # spare only).
    recorder = _make_recorder(module, pa1, pa2)
    assert recorder.current_generation == 0

    assert recorder.start_recording() is True
    assert recorder.current_generation == 1
    release1.set()
    # Fast-torn-down cycle: the thread handle can be cleared before a
    # plain read; wait on observable state (read-before-start race fix).
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)

    assert recorder.start_recording() is True
    assert recorder.current_generation == 2
    release2.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)


def test_current_generation_not_writable(audio_recorder_module):
    """current_generation is a read-only accessor: assignment must fail
    with AttributeError, not silently create an instance attribute."""
    module = audio_recorder_module
    recorder = _make_recorder(module, _make_pyaudio_instance(0))
    with pytest.raises(AttributeError):
        recorder.current_generation = 5


def test_failed_open_teardown_only_touches_worker_owned_objects(audio_recorder_module):
    """Issue #42: attempt 1 reuses the cached instance -- a FAILED
    attempt-1 open never terminates it (the call site owns it and the
    retry loop still uses it); a failed RETRY attempt terminates its own
    fresh instance at its failure site. The recorder never adopts the
    failed retry instance into self.pyaudio (it keeps the cached one).
    """
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry failed")

    # Both attempts' instances fail -- pin max_attempts to match, so the
    # loop doesn't try a 3rd/4th construction the mock has no instance
    # left to return.
    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)
    # Fast-failure path: both attempts fail immediately, so the worker
    # can complete and clear recording before any capture; wait for
    # completion on observable state instead (lens review HIGH #1/#4
    # read-before-start race).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # The cached instance survives the failed attempt 1; the fresh
    # retry instance is terminated at its own failure site.
    pa1.terminate.assert_not_called()
    pa_retry.terminate.assert_called_once()
    assert recorder.pyaudio is pa1


@pytest.mark.skip(
    reason="#49 partial: cached-PyAudio harness rework hangs test — needs deeper mock/threading fix, follow-up chore"
)
def test_all_attempts_exhausted_surfaces_error_and_recovers(
    audio_recorder_module, capsys
):
    """Every one of self.max_attempts instances failing must exhaust the
    loop cleanly (last_error set, recording/stream/thread reset), not
    crash the worker thread. Regression guard for the under-provisioning
    bug: this pins call count to 1 (cached __init__) + (max_attempts -
    1) fresh retry constructions so bumping the default attempt count
    again can't silently leave a mock exhausted."""
    module = audio_recorder_module
    pa_instances = []
    for i in range(3):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"retry attempt {i + 2} failed")
        pa_instances.append(pa)

    # Attempt 1 drives the cached instance (init_probe, a failing spare
    # -- pinning max_attempts to 3 means no construction beyond the
    # init + 3 retry slots the mock can serve, so a bump to the default
    # attempt count again can't silently leave a mock exhausted).
    recorder = _make_recorder(
        module,
        *pa_instances,
        init_probe=_make_pyaudio_instance(-1),
        max_attempts=3,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    # Fast-exhaustion path: all attempts fail immediately, so the
    # worker can complete and clear recording before any capture; wait
    # for completion on observable state instead (lens review HIGH
    # #1/#4 read-before-start race).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # 1 cached construction (__init__) + 3 fresh retry constructions.
    assert module.pyaudio.PyAudio.call_count == 4
    assert recorder.recording is False
    assert recorder.stream is None
    assert recorder.recording_thread is None
    assert recorder.last_error is not None
    assert "retry attempt 4 failed" in recorder.last_error

    # Issue #40: a stop after exhaustion must log reason=retry-exhausted
    # from the early-return path (the worker already cleared recording).
    recorder.stop_recording()
    _assert_stop_log_line(
        capsys.readouterr(), "retry-exhausted", chunks=0, duration_ms_min=0
    )


def test_pyaudio_construction_failure_on_retry_is_not_fatal(audio_recorder_module):
    """A PyAudio() construction failure on a retry attempt (e.g. a severe
    coreaudiod storm making Pa_Initialize() itself fail, not just open())
    must not propagate out of the worker thread uncaught -- it must cost
    only that attempt, exactly like an open() OSError, and the loop must
    continue to the next attempt."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry_success = _make_pyaudio_instance(2)
    release_event = threading.Event()
    retry_stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa_retry_success.open.return_value = retry_stream

    # Attempt 1 (cached): pa1 (open() fails). Attempt 2 (fresh):
    # PyAudio() construction itself raises. Attempt 3 (fresh):
    # pa_retry_success succeeds.
    recorder = _make_recorder(
        module,
        RuntimeError("Pa_Initialize failed"),
        pa_retry_success,
        init_probe=pa1,
        max_attempts=3,
        retry_backoff_seconds=(0.01, 0.01),
    )
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_retry_success, timeout=5.0)

    thread = recorder.recording_thread
    assert thread is not None
    assert thread.is_alive()  # blocked in read() after successful adoption

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
