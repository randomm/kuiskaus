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

import importlib
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


def _make_recorder(module, *pa_instances: MagicMock):
    """Build an AudioRecorder whose successive pyaudio.PyAudio() calls
    return pa_instances in order (constructor call first, retries after)."""
    module.pyaudio.PyAudio = MagicMock(side_effect=list(pa_instances))
    return module.AudioRecorder()


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

    recorder = _make_recorder(module, pa1, pa_retry)
    first_thread = None
    assert recorder.start_recording() is True
    deadline = time.monotonic() + 2.0
    while first_thread is None and time.monotonic() < deadline:
        first_thread = recorder.recording_thread
        if first_thread is None:
            time.sleep(0.01)
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

    recorder = _make_recorder(module, pa1, pa_retry)
    thread = None
    assert recorder.start_recording() is True
    deadline = time.monotonic() + 2.0
    while thread is None and time.monotonic() < deadline:
        thread = recorder.recording_thread
        if thread is None:
            time.sleep(0.01)
    if thread is not None:
        thread.join(timeout=2.0)

    result = recorder.stop_recording()

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.size == 0


def test_failed_open_teardown_only_touches_worker_owned_objects(audio_recorder_module):
    """Teardown after a failed retry must terminate only the fresh
    retry-owned instance, never the original shared self.pyaudio."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)
    pa_retry.open.side_effect = OSError("retry failed")

    recorder = _make_recorder(module, pa1, pa_retry)
    thread = None
    assert recorder.start_recording() is True
    deadline = time.monotonic() + 2.0
    while thread is None and time.monotonic() < deadline:
        thread = recorder.recording_thread
        if thread is None:
            time.sleep(0.01)
    if thread is not None:
        thread.join(timeout=2.0)

    pa_retry.terminate.assert_called_once()
    pa1.terminate.assert_not_called()


# ---------------------------------------------------------------------------
# Retry: inline, best-effort, exactly one
# ---------------------------------------------------------------------------


def test_retry_succeeds_constructs_fresh_pyaudio_and_reresolves_device(
    audio_recorder_module,
):
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
    assert module.pyaudio.PyAudio.call_count == 1  # constructor only, so far

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    # The retry (a second PyAudio construction) happens before the read
    # loop starts, so once the worker reaches read() the adoption is
    # complete. Bounded wait for the retry construction: the test cannot
    # assume the worker has finished the retry by the time it returns.
    deadline = time.monotonic() + 2.0
    while module.pyaudio.PyAudio.call_count < 2:
        assert time.monotonic() < deadline, "retry PyAudio never constructed"
        time.sleep(0.01)
    # Adoption precedes the read loop, so by the time the worker is
    # blocked in read() the retry instance is authoritative.
    pa1.get_default_input_device_info.assert_called_once()  # constructor only
    pa_retry.get_default_input_device_info.assert_called_once()  # re-resolved
    assert recorder.pyaudio is pa_retry
    assert recorder.input_device_index == 7
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
    deadline = time.monotonic() + 2.0
    while thread is None and time.monotonic() < deadline:
        thread = recorder.recording_thread
        if thread is None:
            time.sleep(0.01)
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


def test_happy_path_uses_existing_pyaudio_without_retry(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, pa1)
    assert module.pyaudio.PyAudio.call_count == 1
    original_pyaudio = recorder.pyaudio
    original_device_index = recorder.input_device_index

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread.is_alive()  # worker is blocked in read()

    # No construction, no reassignment, no re-resolution on the happy
    # path -- asserted while the worker is deterministically alive.
    assert module.pyaudio.PyAudio.call_count == 1
    pa1.get_default_input_device_info.assert_called_once()  # constructor only
    assert recorder.pyaudio is original_pyaudio
    assert recorder.input_device_index == original_device_index
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

    recorder = _make_recorder(module, pa1, pa_retry)
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

    assert elapsed < 5.0  # join(timeout=1.0) must actually bound this
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
    """A stuck worker that outlives stop_recording()'s join(timeout=1.0)
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

    # stop_recording()'s join(timeout=1.0) expires; the worker is still
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
