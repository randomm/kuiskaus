"""Hardware-free unit tests for AudioRecorder (issue #16).

A failed microphone open must cost the user one recording, visibly -- not
the session. These tests exercise the concurrency guard (lock + liveness
aware admission + stale-state recovery + generation counter), the inline
retry, and last_error surfacing without ever touching real hardware.

pyaudio is stubbed per-test via monkeypatch.setitem + importlib.reload
(never at module scope -- a module-scope sys.modules stub would leak
across the shared pytest process, see #22). tests/test_postprocessor.py
and tests/test_callback_dispatcher.py are the leak-free references.

Worker-lifecycle control: every test that needs to observe live worker
state (or whose worker could otherwise race to completion and clear
recording_thread to None before the test reads it) blocks the mocked
stream.read() on a test-owned threading.Event. The test captures
recorder.recording_thread while the worker is provably still alive,
asserts on live state, then sets the event and joins with a bounded
timeout. No wait in this file is unbounded.
"""

import gc
import importlib
import re
import sys
import threading
import time
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest


def _cleanup_recorder(recorder):
    """Deterministically tear down a test recorder's worker.

    AudioRecorder.__del__ calls cleanup(), which does thread.join() with
    NO timeout. At interpreter shutdown that join never returns while
    any other thread (pytest's own, a leaked daemon, ...) is alive, and
    the whole test process hangs at exit -- the same false-confidence
    failure class issue #16 eliminates, relocated to the test harness.
    Calling cleanup() explicitly here while the worker is (or is about to
    be) finished makes the GC-time path a no-op. Bounded: if the worker
    is stuck in open() past stop_recording()'s own 1s join, cleanup()
    skips the native terminate() rather than blocking.
    """
    try:
        recorder.cleanup()
    except OSError:
        print("Error closing stream: recorder teardown in test")


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


def _make_pyaudio_instance(index: int = 0) -> MagicMock:
    """A mock PyAudio() instance with a resolvable default input device."""
    instance = MagicMock(name=f"PyAudioInstance-{index}")
    instance.get_default_input_device_info.return_value = {"index": index}
    return instance


def _make_recorder(module, *pa_instances: MagicMock | BaseException, **kwargs):
    """Build an AudioRecorder whose successive pyaudio.PyAudio() calls
    return pa_instances in order across every retry-loop attempt of every
    recording this recorder makes (attempt 1 first, retries after).

    __init__ (issue #37 task-c) constructs and immediately terminates its
    own temporary device-probe PyAudio() instance before any of
    ``pa_instances`` is consumed -- that construction is synthesized
    internally here so callers only need to describe the construction
    sequence ``_open_stream_with_retry`` will see; it never has to
    account for __init__'s probe.

    An ``Exception`` instance in ``pa_instances`` is raised instead of
    returned (mock's own ``side_effect``-iterable behaviour), simulating
    a PyAudio() construction failure on that attempt.
    """
    init_probe = _make_pyaudio_instance(-1)  # -1: __init__'s own device-probe instance
    module.pyaudio.PyAudio = MagicMock(side_effect=[init_probe, *pa_instances])
    return module.AudioRecorder(**kwargs)


def _blocking_stream(read_error: Exception, release_event: threading.Event):
    """A mock stream whose read() blocks on a test-owned event, then
    raises read_error.

    This is the synchronisation pattern for controlling the worker:
    while release_event is unset the worker is deterministically alive
    (blocked inside read()), so recorder.recording_thread cannot have
    been cleared by worker teardown yet. The test sets release_event to
    end the recording loop and joins the captured thread.
    """

    def fake_read(*_args, **_kwargs):
        release_event.wait(timeout=10.0)
        raise read_error

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    return stream


def _wait_until(
    predicate,
    timeout: float = 2.0,
    poll: float = 0.01,
) -> bool:
    """Poll until predicate() is truthy. True on success, False on timeout.

    The poll delay uses ``threading.Event().wait()`` rather than
    ``time.sleep``: tests that monkeypatch ``module.time.sleep`` (which
    also patches this file's time.sleep, since both resolve to the same
    stdlib module object) would otherwise no-op the poll and busy-loop
    against the patched mock -- the old inline _real_wait loops had the
    same immune delay, which this helper now centralises.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        threading.Event().wait(timeout=poll)
    return predicate()


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

    # Both mocked PyAudio() instances fail -- pin max_attempts to match,
    # so the loop doesn't try a 3rd/4th construction the mock has no
    # instance left to return.
    recorder = _make_recorder(module, pa1, pa_retry, max_attempts=2)
    first_thread = None
    assert recorder.start_recording() is True
    _wait_until(lambda: recorder.recording_thread is not None)
    first_thread = recorder.recording_thread
    # Fast-failure path: the worker sets recording_thread inside
    # start_recording() and clears it in teardown; both are under the
    # same lock, so either the worker is still running (we captured a
    # live thread) or it already finished and cleared the state (which
    # is the subject of the assertions below). Polling can miss the
    # thread entirely -- that is the expected fast-failure outcome, not
    # a failure of the test.
    if first_thread is not None:
        first_thread.join(timeout=2.0)

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
    assert second_thread is not first_thread
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

    # Both mocked PyAudio() instances fail -- pin max_attempts to match,
    # so the loop doesn't try a 3rd/4th construction the mock has no
    # instance left to return.
    recorder = _make_recorder(module, pa1, pa_retry, max_attempts=2)
    thread = None
    assert recorder.start_recording() is True
    _wait_until(lambda: recorder.recording_thread is not None)
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=2.0)

    result = recorder.stop_recording()

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.size == 0


def test_failed_open_teardown_only_touches_worker_owned_objects(audio_recorder_module):
    """Every attempt -- including attempt 1 -- constructs its own fresh,
    disposable PyAudio() instance (issue #37 task-c); a failed attempt
    owns no stream, so terminating it directly at its own failure site is
    always safe. Both attempts' instances get terminated on their own
    failure; the recorder never adopts either into self.pyaudio."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry failed")

    # Both mocked PyAudio() instances fail -- pin max_attempts to match,
    # so the loop doesn't try a 3rd/4th construction the mock has no
    # instance left to return.
    recorder = _make_recorder(module, pa1, pa_retry, max_attempts=2)
    thread = None
    assert recorder.start_recording() is True
    _wait_until(lambda: recorder.recording_thread is not None)
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=2.0)

    pa1.terminate.assert_called_once()
    pa_retry.terminate.assert_called_once()
    assert recorder.pyaudio is None


def test_all_attempts_exhausted_surfaces_error_and_recovers(audio_recorder_module):
    """Every one of self.max_attempts instances failing must exhaust the
    loop cleanly (last_error set, recording/stream/thread reset), not
    crash the worker thread. Regression guard for the under-provisioning
    bug: this pins call count to max_attempts so bumping the default
    attempt count again can't silently leave a mock exhausted."""
    module = audio_recorder_module
    pa_instances = []
    for i in range(4):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"attempt {i + 1} failed")
        pa_instances.append(pa)

    recorder = _make_recorder(
        module,
        *pa_instances,
        max_attempts=4,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    thread = None
    assert recorder.start_recording() is True
    _wait_until(lambda: recorder.recording_thread is not None, timeout=5.0)
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    # +1 for __init__'s own device-probe construction (issue #37 task-c),
    # synthesized by _make_recorder ahead of pa_instances.
    assert module.pyaudio.PyAudio.call_count == 5
    assert recorder.recording is False
    assert recorder.stream is None
    assert recorder.recording_thread is None
    assert recorder.last_error is not None
    assert "attempt 4 failed" in recorder.last_error


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

    # Attempt 1: pa1 (open() fails). Attempt 2: PyAudio() construction
    # itself raises. Attempt 3: pa_retry_success succeeds.
    recorder = _make_recorder(
        module,
        pa1,
        RuntimeError("Pa_Initialize failed"),
        pa_retry_success,
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


# ---------------------------------------------------------------------------
# Retry: inline, best-effort, exactly one
# ---------------------------------------------------------------------------


def test_retry_succeeds_constructs_fresh_pyaudio_and_reresolves_device(
    audio_recorder_module,
):
    """Reshaped for issue #37 task-c: attempt 1 no longer reuses a cached
    self.pyaudio/input_device_index (that cache is retired) -- every
    attempt, including attempt 1, constructs its own fresh PyAudio() and
    re-resolves the device. A successful retry can land on any attempt
    from 2 up to self.max_attempts, so this asserts eventual adoption
    with a loosened attempt-count bound rather than an exact count."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(7)
    release_event = threading.Event()
    retry_stream = _blocking_stream(
        OSError("stop the loop after adoption"), release_event
    )
    pa_retry.open.return_value = retry_stream

    recorder = _make_recorder(module, pa1, pa_retry)
    # +1 for __init__'s own device-probe construction (issue #37 task-c).
    assert module.pyaudio.PyAudio.call_count == 1

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None

    # Bounded wait for adoption: recorder.pyaudio only becomes pa_retry
    # once the retry's open() has succeeded and been adopted.
    assert _wait_until(lambda: recorder.pyaudio is pa_retry)

    # attempts_taken excludes __init__'s probe construction. At least 2
    # (attempt 1 with pa1 always fails first in this test), at most
    # max_attempts (the loop's own bound).
    attempts_taken = module.pyaudio.PyAudio.call_count - 1
    assert 2 <= attempts_taken <= recorder.max_attempts
    pa1.get_default_input_device_info.assert_called_once()  # attempt 1
    pa_retry.get_default_input_device_info.assert_called_once()  # re-resolved
    assert thread.is_alive()  # still blocked in read()

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    assert recorder.last_error is None
    retry_stream.stop_stream.assert_called_once()
    retry_stream.close.assert_called_once()


def test_retry_runtime_error_from_device_lookup_sets_last_error(
    audio_recorder_module,
):
    """Round-2 review CRITICAL finding: if the retry's own
    _find_default_input_device() call exhausts its fallback loop and
    raises RuntimeError (no input device at all -- e.g. the mic was
    unplugged between the two attempts), the worker must not die
    silently. Previously only OSError was caught around the retry, so
    this RuntimeError propagated out of _recording_worker and killed the
    daemon thread with recording still True and last_error still None."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.get_default_input_device_info.side_effect = OSError("no default device")
    pa_retry.get_device_count.return_value = 0  # fallback loop finds nothing

    recorder = _make_recorder(module, pa1, pa_retry)
    thread = None
    assert recorder.start_recording() is True
    _wait_until(lambda: recorder.recording_thread is not None)
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=2.0)

    assert recorder.recording is False
    assert recorder.stream is None
    assert recorder.recording_thread is None
    assert recorder.last_error is not None
    pa_retry.open.assert_not_called()


def test_late_successful_open_does_not_clobber_already_surfaced_last_error(
    audio_recorder_module,
):
    """Round-2 review ISSUES finding: if stop_recording()'s stuck-open
    detection already set last_error for this generation (worker still
    blocked in open() past the join timeout) and the worker's open()
    subsequently succeeds late, it must not clear that already-surfaced
    error -- self.recording is already False for this generation, so
    there is no active session left to clear it for."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    stream = MagicMock(name="late-success-stream")
    pa1.open.return_value = stream

    recorder = _make_recorder(module, pa1)
    recorder._generation = 1
    recorder.recording = False  # session already ended via stop_recording()
    recorder.last_error = "microphone busy \u2014 recording did not start"

    recorder._recording_worker(1)

    assert recorder.last_error == "microphone busy \u2014 recording did not start"
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()


def test_first_attempt_constructs_fresh_pyaudio_and_resolves_device(
    audio_recorder_module,
):
    """Reshaped for issue #37 task-c (renamed from
    test_happy_path_uses_existing_pyaudio_without_retry): __init__ no
    longer caches a long-lived self.pyaudio -- self.pyaudio is None until
    the first successful open(). Attempt 1 constructs its own fresh
    PyAudio() and resolves the device on it, exactly like every other
    attempt, rather than reusing a cached instance."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, pa1)
    # 1: __init__'s own device-probe construction only -- self.pyaudio is
    # still None, no attempt has run yet.
    assert module.pyaudio.PyAudio.call_count == 1
    assert recorder.pyaudio is None

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread.is_alive()  # worker is blocked in read()

    # Fresh construction on attempt 1: probe (1) + attempt 1 (1) = 2,
    # asserted while the worker is deterministically alive.
    assert module.pyaudio.PyAudio.call_count == 2
    pa1.get_default_input_device_info.assert_called_once()  # attempt 1's own resolution
    assert recorder.pyaudio is pa1
    assert recorder.last_error is None

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_worker_thread_is_daemon(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread.daemon is True

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_cleanup_skips_terminate_when_worker_still_alive(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()

    def blocking_open(**_kwargs):
        release_event.wait(timeout=10.0)
        raise OSError("released")

    pa1.open.side_effect = blocking_open
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry fails")

    recorder = _make_recorder(module, pa1, pa_retry)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread

    # The worker is deterministically alive (blocked in open()); this is
    # the exact precondition cleanup() guards against.
    assert thread.is_alive()
    recorder.cleanup()

    pa1.terminate.assert_not_called()

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Concurrency guard: liveness-aware admission, stale recovery, generation
# ---------------------------------------------------------------------------


def test_admission_refuses_while_worker_alive(audio_recorder_module):
    """A bare boolean admission guard would also refuse here -- this test
    alone doesn't prove liveness-awareness, see the stale-recovery test
    below for the case a bare boolean gets wrong."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()

    def blocking_open(**_kwargs):
        release_event.wait(timeout=10.0)
        raise OSError("released")

    pa1.open.side_effect = blocking_open
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry fails too")

    # Both mocked PyAudio() instances fail -- pin max_attempts to match,
    # so the loop doesn't try a 3rd/4th construction the mock has no
    # instance left to return.
    recorder = _make_recorder(module, pa1, pa_retry, max_attempts=2)
    assert recorder.start_recording() is True
    first_thread = recorder.recording_thread

    assert recorder.start_recording() is False
    assert recorder.recording_thread is first_thread  # no second worker spawned

    release_event.set()
    first_thread.join(timeout=10.0)
    assert not first_thread.is_alive()


def test_stale_state_recovery_when_recording_true_but_thread_dead(
    audio_recorder_module,
):
    """recording=True with a dead thread must NOT wedge -- a bare boolean
    guard (`if not self.recording`) would refuse forever here."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, pa1)

    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join(timeout=2.0)
    assert not dead_thread.is_alive()

    recorder.recording = True
    recorder.recording_thread = dead_thread

    assert recorder.start_recording() is True
    new_thread = recorder.recording_thread
    assert new_thread is not dead_thread
    assert new_thread.is_alive()  # blocked in read()

    release_event.set()
    new_thread.join(timeout=10.0)
    assert not new_thread.is_alive()


def test_release_during_backoff_sleep_aborts_before_next_attempt(
    audio_recorder_module, monkeypatch
):
    """Releasing the hotkey while asleep between backoff attempts must
    abort immediately after waking -- before constructing the next
    PyAudio() instance or calling open() on it -- not just before the
    sleep. Without a post-sleep recheck, a release landing exactly during
    the sleep still burns a full abandoned attempt's worth of PortAudio
    work before the pre-sleep check on the following iteration notices.
    """
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)

    recorder = _make_recorder(module, pa1, pa_retry, max_attempts=2)

    def fake_sleep(_seconds):
        # Simulate the hotkey being released while the worker is asleep
        # between attempts.
        recorder.recording = False

    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    # The abort must land before the retry attempt does any work: probe
    # (1) + attempt 1 (pa1, 1) = 2, no third (retry) construction, no
    # open() call on the retry instance.
    assert module.pyaudio.PyAudio.call_count == 2
    pa_retry.open.assert_not_called()


def test_stale_worker_open_success_does_not_clobber_newer_generation(
    audio_recorder_module,
):
    """A superseded worker whose (late) open() succeeds must not adopt its
    stream into a newer generation's state."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    stale_stream = MagicMock(name="stale-stream")
    pa1.open.return_value = stale_stream

    recorder = _make_recorder(module, pa1)
    recorder._generation = 1
    recorder.recording = True
    recorder.stream = "newer-stream-sentinel"
    recorder.recording_thread = "newer-thread-sentinel"

    # A stale worker for a generation that has already been superseded.
    recorder._recording_worker(0)

    assert recorder.recording is True
    assert recorder.stream == "newer-stream-sentinel"
    assert recorder.recording_thread == "newer-thread-sentinel"
    # The stale worker must still close the stream it opened -- a resource
    # it owns regardless of whether its generation is current.
    stale_stream.stop_stream.assert_called_once()
    stale_stream.close.assert_called_once()

    # Reset the sentinel state so GC-time cleanup() doesn't try to join()
    # a plain string.
    recorder.recording = False
    recorder.recording_thread = None
    recorder.stream = None


def test_stale_worker_post_loop_teardown_does_not_clobber_newer_generation(
    audio_recorder_module,
):
    """A stale worker's post-loop teardown must not null a newer
    generation's self.stream mid-recording."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    stream = MagicMock(name="stream")

    def fake_read(*_args, **_kwargs):
        # Simulate a newer recording taking over while this worker's read
        # loop is running, then let this worker's loop exit naturally.
        recorder._generation = 2
        recorder.recording = True
        recorder.stream = "newer-stream-sentinel"
        recorder.recording_thread = "newer-thread-sentinel"
        raise OSError("stop this worker's loop")

    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, pa1)
    recorder._generation = 1
    recorder.recording = True

    recorder._recording_worker(1)

    assert recorder.recording is True
    assert recorder.stream == "newer-stream-sentinel"
    assert recorder.recording_thread == "newer-thread-sentinel"
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()

    # Reset the sentinel state so GC-time cleanup() doesn't try to join()
    # a plain string.
    recorder.recording = False
    recorder.recording_thread = None
    recorder.stream = None


# ---------------------------------------------------------------------------
# retry_backoff_seconds validation
# ---------------------------------------------------------------------------


def test_empty_retry_backoff_seconds_raises_value_error(audio_recorder_module):
    module = audio_recorder_module
    with pytest.raises(ValueError, match="empty"):
        module.AudioRecorder(retry_backoff_seconds=())


def test_negative_retry_backoff_seconds_raises_value_error(audio_recorder_module):
    module = audio_recorder_module
    with pytest.raises(ValueError, match="negative"):
        module.AudioRecorder(retry_backoff_seconds=(0.1, -1.0))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_retry_backoff_seconds_raises_value_error(
    audio_recorder_module, bad_value
):
    module = audio_recorder_module
    with pytest.raises(ValueError, match="finite"):
        module.AudioRecorder(retry_backoff_seconds=(0.1, bad_value))


def test_invalid_retry_backoff_seconds_does_not_crash_on_gc_cleanup(
    audio_recorder_module,
):
    """A __init__ that raises mid-construction must not blow up in
    cleanup()/__del__ when the partially-constructed instance is
    garbage-collected. __init__ sets the defensive state attributes
    (recording/recording_thread/stream/pyaudio/lock) BEFORE any
    validation can raise, so cleanup() needs no hasattr guard."""
    module = audio_recorder_module
    with pytest.raises(ValueError):
        module.AudioRecorder(retry_backoff_seconds=())
    # No exception raised here means __del__ -> cleanup() handled the
    # partially-constructed instance gracefully.
    gc.collect()


@pytest.mark.parametrize("bad_attempts", [0, -1])
def test_max_attempts_less_than_one_raises_value_error(
    audio_recorder_module, bad_attempts
):
    """max_attempts <= 0 would otherwise silently skip the retry loop and
    surface a bogus "unknown error" (lens review MEDIUM #2)."""
    module = audio_recorder_module
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        module.AudioRecorder(max_attempts=bad_attempts)


def test_oserror_from_device_enumeration_is_retryable(audio_recorder_module):
    """A coreaudiod storm can make device enumeration itself raise OSError
    -9986 (pa.get_device_count()/get_device_info_by_index), not just
    open() (lens review HIGH #3). That must cost one attempt and let the
    loop continue, not propagate out of _recording_worker and kill the
    thread silently (the exact bug #16 this PR closes)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    # Default-info lookup raises (so the fallback enumeration path runs),
    # and the first enumeration pass itself raises OSError -9986.
    pa1.get_default_input_device_info.side_effect = OSError("no default")
    pa1.get_device_count.side_effect = [
        OSError("paInternalError during enumeration"),
        1,
    ]
    pa1.get_device_info_by_index.return_value = {
        "maxInputChannels": 0,
        "maxOutputChannels": 2,
    }
    pa1.open.side_effect = OSError("attempt 1 failed")

    pa2 = _make_pyaudio_instance(1)
    release_event = threading.Event()
    pa2.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)

    recorder = _make_recorder(
        module, pa1, pa2, max_attempts=2, retry_backoff_seconds=(0.01,)
    )
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa2, timeout=5.0)

    thread = recorder.recording_thread
    assert thread is not None
    assert thread.is_alive()  # worker survived the enumeration OSError
    assert recorder.last_error is None

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Stuck-open detection
# ---------------------------------------------------------------------------


def test_stuck_open_sets_last_error_on_join_timeout(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    never_release = threading.Event()

    def stuck_open(**_kwargs):
        never_release.wait(timeout=10.0)
        raise OSError("released late")

    pa1.open.side_effect = stuck_open
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry fails too")

    recorder = _make_recorder(module, pa1, pa_retry)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread

    start = time.monotonic()
    result = recorder.stop_recording()
    elapsed = time.monotonic() - start

    # join(timeout=recorder._stuck_open_timeout_seconds) must actually
    # bound this wait -- not the raw stuck-worker duration (never_release
    # blocks for up to 10s). Generous slack for scheduling jitter only.
    assert elapsed < recorder._stuck_open_timeout_seconds + 2.0
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    assert recorder.last_error is not None
    assert "busy" in recorder.last_error.lower()

    never_release.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_admission_refuses_orphaned_worker_still_alive_after_stuck_stop(
    audio_recorder_module,
):
    """A stuck worker that outlives stop_recording()'s join(timeout=...)
    clears self.recording to False (issue #16 regression review, round 1)
    but must still block a second worker from calling pyaudio.open() on
    the same shared self.pyaudio while it is physically still running --
    otherwise two threads could race into a native open() call
    concurrently. Liveness of recording_thread, not self.recording, is
    the correct admission signal for this orphaned-but-alive case."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    never_release = threading.Event()

    def stuck_open(**_kwargs):
        never_release.wait(timeout=10.0)
        raise OSError("released late")

    pa1.open.side_effect = stuck_open
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry fails too")

    recorder = _make_recorder(module, pa1, pa_retry)
    assert recorder.start_recording() is True
    stuck_thread = recorder.recording_thread

    # stop_recording()'s join(timeout=...) expires; the worker is still
    # alive, blocked inside pyaudio.open().
    recorder.stop_recording()
    assert recorder.recording is False  # release path already cleared this
    assert stuck_thread.is_alive()

    # A second press must be refused -- the orphaned worker may still be
    # about to call pyaudio.open() on self.pyaudio.
    assert recorder.start_recording() is False
    assert recorder.recording_thread is stuck_thread

    # Once the native call finally returns and the worker tears itself
    # down, admission self-heals without needing a restart.
    never_release.set()
    stuck_thread.join(timeout=10.0)
    assert not stuck_thread.is_alive()

    # Recovery is observable live: start with a blocking read so the new
    # worker cannot tear itself down before we assert.
    post_recovery_release = threading.Event()
    pa1.open.side_effect = None
    pa1.open.return_value = _blocking_stream(
        OSError("stop the loop"), post_recovery_release
    )
    assert recorder.start_recording() is True
    post_recovery_thread = recorder.recording_thread
    assert post_recovery_thread.is_alive()
    assert post_recovery_thread is not stuck_thread

    post_recovery_release.set()
    post_recovery_thread.join(timeout=10.0)
    assert not post_recovery_thread.is_alive()


# ---------------------------------------------------------------------------
# Backoff loop cadence and exhaustion (issue #37 task-e)
# ---------------------------------------------------------------------------


def test_backoff_loop_makes_up_to_n_attempts_with_sleep_between(
    audio_recorder_module, monkeypatch
):
    """max_attempts consecutive OSErrors exhaust the loop, constructing
    exactly one fresh PyAudio() per attempt and sleeping exactly
    max_attempts - 1 times between them."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa_instances = []
    for i in range(4):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"attempt {i + 1} failed")
        pa_instances.append(pa)

    recorder = _make_recorder(
        module,
        *pa_instances,
        max_attempts=4,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    # -1 for __init__'s own device-probe construction (issue #37 task-c).
    assert module.pyaudio.PyAudio.call_count - 1 == 4
    assert module.time.sleep.call_count == 3


def test_backoff_sleep_cadence_matches_schedule(audio_recorder_module, monkeypatch):
    """The sleep durations actually used match the effective backoff
    schedule verbatim, in order."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa_instances = []
    for i in range(4):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"attempt {i + 1} failed")
        pa_instances.append(pa)

    schedule = (0.11, 0.22, 0.33)
    recorder = _make_recorder(
        module, *pa_instances, max_attempts=4, retry_backoff_seconds=schedule
    )
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert module.time.sleep.call_args_list == [mock.call(s) for s in schedule]


def test_backoff_loop_exhausts_after_max_attempts_sets_last_error(
    audio_recorder_module, monkeypatch
):
    """Exhausting every one of max_attempts attempts sets last_error and
    resets recording/stream/thread state cleanly."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa_instances = []
    for i in range(4):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"attempt {i + 1} failed")
        pa_instances.append(pa)

    recorder = _make_recorder(
        module,
        *pa_instances,
        max_attempts=4,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert recorder.recording is False
    assert recorder.stream is None
    assert recorder.recording_thread is None
    assert recorder.last_error is not None
    assert "attempt 4 failed" in recorder.last_error


def test_backoff_loop_succeeds_on_middle_attempt(audio_recorder_module, monkeypatch):
    """Attempts 1 and 2 fail, attempt 3 succeeds: only 2 sleeps occur and
    last_error is left None."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("attempt 1 failed")
    pa2 = _make_pyaudio_instance(1)
    pa2.open.side_effect = OSError("attempt 2 failed")
    pa3 = _make_pyaudio_instance(2)
    release_event = threading.Event()
    pa3.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)

    recorder = _make_recorder(
        module,
        pa1,
        pa2,
        pa3,
        max_attempts=4,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None

    assert _wait_until(lambda: recorder.pyaudio is pa3)
    assert module.time.sleep.call_count == 2
    assert recorder.last_error is None

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# last_error message content (issue #37 task-e)
# ---------------------------------------------------------------------------


def test_last_error_mentions_killall_coreaudiod_for_paInternalError(
    audio_recorder_module,
):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    err = OSError("Internal PortAudio error")
    err.errno = module.PA_INTERNAL_ERROR_ERRNO
    pa1.open.side_effect = err

    recorder = _make_recorder(module, pa1, max_attempts=1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    assert recorder.last_error is not None
    assert "sudo killall coreaudiod" in recorder.last_error


def test_last_error_generic_for_non_paInternalError(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    err = OSError("Device unavailable")
    err.errno = -9985  # paDeviceUnavailable, deliberately not -9986
    pa1.open.side_effect = err

    recorder = _make_recorder(module, pa1, max_attempts=1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    assert recorder.last_error is not None
    assert "killall" not in recorder.last_error
    assert "Microphone unavailable" in recorder.last_error


# ---------------------------------------------------------------------------
# PyAudio ownership (issue #37 task-c/e)
# ---------------------------------------------------------------------------


def test_fresh_pyaudio_constructed_every_start_recording_call(audio_recorder_module):
    """Two consecutive recordings each construct and adopt their own
    fresh PyAudio() instance -- no cross-recording caching."""
    module = audio_recorder_module
    pa_first = _make_pyaudio_instance(0)
    first_release = threading.Event()
    pa_first.open.return_value = _blocking_stream(
        OSError("stop first recording"), first_release
    )
    pa_second = _make_pyaudio_instance(1)

    recorder = _make_recorder(module, pa_first, pa_second)

    assert recorder.start_recording() is True
    first_thread = recorder.recording_thread
    assert first_thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa_first)

    first_release.set()
    first_thread.join(timeout=10.0)
    assert not first_thread.is_alive()

    second_release = threading.Event()
    pa_second.open.return_value = _blocking_stream(
        OSError("stop second recording"), second_release
    )
    assert recorder.start_recording() is True
    second_thread = recorder.recording_thread
    assert second_thread is not None
    assert second_thread is not first_thread
    assert _wait_until(lambda: recorder.pyaudio is pa_second)
    assert recorder.pyaudio is not pa_first

    # 1 probe (init) + 1 (first recording's attempt 1) + 1 (second
    # recording's attempt 1) == 3.
    assert module.pyaudio.PyAudio.call_count == 3

    second_release.set()
    second_thread.join(timeout=10.0)
    assert not second_thread.is_alive()


def test_previous_pyaudio_terminated_when_stream_none_before_next_recording(
    audio_recorder_module,
):
    """Between recordings, once the first recording's worker has fully
    torn down (self.stream is None -- the invariant every teardown path
    maintains), the second recording's successful attempt 1 disposes of
    the first recording's PyAudio() via lock-scoped local-capture."""
    module = audio_recorder_module
    pa_first = _make_pyaudio_instance(0)
    first_release = threading.Event()
    pa_first.open.return_value = _blocking_stream(
        OSError("stop first recording"), first_release
    )
    pa_second = _make_pyaudio_instance(1)

    recorder = _make_recorder(module, pa_first, pa_second)

    assert recorder.start_recording() is True
    first_thread = recorder.recording_thread
    assert first_thread is not None
    deadline = time.monotonic() + 2.0
    while recorder.pyaudio is not pa_first and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder.pyaudio is pa_first

    first_release.set()
    first_thread.join(timeout=10.0)
    assert not first_thread.is_alive()
    assert recorder.stream is None  # teardown invariant

    second_release = threading.Event()
    pa_second.open.return_value = _blocking_stream(
        OSError("stop second recording"), second_release
    )
    assert recorder.start_recording() is True
    second_thread = recorder.recording_thread
    assert second_thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa_second)

    pa_first.terminate.assert_called_once()

    second_release.set()
    second_thread.join(timeout=10.0)
    assert not second_thread.is_alive()


def test_previous_pyaudio_not_terminated_when_stream_not_none(audio_recorder_module):
    """Negative case / defensive guard: if self.stream were somehow not
    None at adoption time (the invariant that should always hold on
    every real teardown path), the old PyAudio() instance must NOT be
    terminated -- a live stream might still depend on it."""
    module = audio_recorder_module
    old_pa = _make_pyaudio_instance(0)
    new_pa = _make_pyaudio_instance(1)
    new_pa.open.return_value = MagicMock(name="new-stream")

    recorder = _make_recorder(module, new_pa)
    recorder.pyaudio = old_pa
    recorder.stream = MagicMock(name="still-live-stream-sentinel")
    recorder.recording = True

    result = recorder._open_stream_with_retry(recorder._generation)

    assert result is not None
    assert recorder.pyaudio is new_pa
    old_pa.terminate.assert_not_called()

    # Reset so GC-time cleanup() doesn't try to close/join sentinel state.
    recorder.recording = False
    recorder.stream = None


# ---------------------------------------------------------------------------
# Mid-backoff abort guards (issue #37 task-e)
# ---------------------------------------------------------------------------


def test_backoff_loop_aborts_on_generation_supersede(
    audio_recorder_module, monkeypatch
):
    """A generation bump mid-backoff (a newer recording superseding this
    one) aborts the loop before the next attempt is even constructed."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)

    recorder = _make_recorder(module, pa1, pa_retry, max_attempts=2)

    def fake_sleep(_seconds):
        # Simulate a newer recording's generation superseding this one
        # while asleep between attempts.
        recorder._generation += 1

    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    # The abort must land before the retry attempt does any work: probe
    # (1) + attempt 1 (pa1, 1) = 2, no second (retry) construction.
    assert module.pyaudio.PyAudio.call_count == 2
    pa_retry.open.assert_not_called()
    assert recorder.last_error is None  # abort path writes no state


def test_backoff_loop_aborts_on_release_mid_backoff(audio_recorder_module, monkeypatch):
    """Releasing mid-backoff (self.recording set False under lock between
    attempts) aborts before the next attempt's sleep/construction and
    leaves last_error untouched -- the abort path returns without
    writing any state."""
    module = audio_recorder_module
    sleep_mock = MagicMock()
    monkeypatch.setattr(module.time, "sleep", sleep_mock)

    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("attempt 1 failed")
    pa2 = _make_pyaudio_instance(1)
    pa2.open.side_effect = OSError("attempt 2 failed")
    pa3 = _make_pyaudio_instance(2)

    recorder = _make_recorder(
        module, pa1, pa2, pa3, max_attempts=3, retry_backoff_seconds=(0.01, 0.01)
    )

    def release_after_first_sleep(_seconds):
        recorder.recording = False

    sleep_mock.side_effect = release_after_first_sleep

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    # Only attempt 1's failure triggers the first sleep; the loop aborts
    # right after waking, before attempt 2 is constructed or a second
    # sleep is scheduled.
    assert sleep_mock.call_count == 1
    assert module.pyaudio.PyAudio.call_count == 2  # probe + attempt 1 only
    pa2.open.assert_not_called()
    pa3.open.assert_not_called()
    assert recorder.last_error is None


# ---------------------------------------------------------------------------
# Structured per-attempt logging (issue #37 task-e)
# ---------------------------------------------------------------------------


def test_per_attempt_log_line_emitted_with_expected_fields(
    audio_recorder_module, capsys
):
    """Every attempt in the retry loop emits exactly one structured
    [audio.retry] log line matching the documented field format."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    err = OSError("Internal PortAudio error")
    err.errno = module.PA_INTERNAL_ERROR_ERRNO
    pa1.open.side_effect = err
    pa_retry = _make_pyaudio_instance(1)
    release_event = threading.Event()
    pa_retry.open.return_value = _blocking_stream(
        OSError("stop the loop"), release_event
    )

    recorder = _make_recorder(
        module, pa1, pa_retry, max_attempts=2, retry_backoff_seconds=(0.01,)
    )
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None

    assert _wait_until(lambda: recorder.pyaudio is pa_retry)

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    captured = capsys.readouterr()
    log_lines = [
        line for line in captured.out.splitlines() if line.startswith("[audio.retry]")
    ]
    pattern = re.compile(
        r"^\[audio\.retry\] attempt=(\d+)/(\d+) elapsed_ms=(\d+) "
        r"errno=(-?\d+|-) action=(sleep|open|adopt|abort)$"
    )
    assert len(log_lines) >= 3  # attempt 1 open-fail, sleep, attempt 2 adopt
    for line in log_lines:
        assert pattern.match(line), line
    assert any(
        "action=open" in line and f"errno={module.PA_INTERNAL_ERROR_ERRNO}" in line
        for line in log_lines
    )
    assert any("action=sleep" in line for line in log_lines)
    assert any("action=adopt" in line for line in log_lines)


# ---------------------------------------------------------------------------
# Constructor kwargs (issue #37 task-e)
# ---------------------------------------------------------------------------


def test_constructor_kwargs_override_module_defaults(audio_recorder_module):
    """max_attempts and retry_backoff_seconds constructor kwargs override
    the module-level defaults, and _stuck_open_timeout_seconds is derived
    from the effective (kwarg) values, not the module constants."""
    module = audio_recorder_module
    recorder = _make_recorder(module, max_attempts=2, retry_backoff_seconds=(0.05,))

    assert recorder.max_attempts == 2
    assert recorder.retry_backoff_seconds == (0.05,)
    assert recorder._stuck_open_timeout_seconds == pytest.approx(0.55)
    assert recorder.max_attempts != module.MAX_ATTEMPTS
    assert recorder.retry_backoff_seconds != module.RETRY_BACKOFF_SECONDS
