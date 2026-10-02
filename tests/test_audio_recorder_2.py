"""test_audio_recorder tests, part 2 of 6 (issue #70)."""

import gc
import threading
from unittest.mock import MagicMock

import pytest

from tests.support_audio_recorder import (
    _blocking_stream,
    _make_pyaudio_instance,
    _make_recorder,
    _wait_until,
)


# ---------------------------------------------------------------------------
# Retry: inline, best-effort, exactly one
# ---------------------------------------------------------------------------
def test_retry_succeeds_constructs_fresh_pyaudio_and_reresolves_device(
    audio_recorder_module,
):
    """Issue #42: attempt 1 reuses the cached self.pyaudio (pa1) and
    its cached device index; a failed attempt-1 open costs no
    construction. The retry (attempt 2) constructs a fresh PyAudio()
    and re-resolves the device against it -- PR #38's retry behaviour,
    unchanged. Success lands on exactly attempt 2: 1 cached
    construction (__init__) + 1 fresh retry = 2 total."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(7)
    release_event = threading.Event()
    retry_stream = _blocking_stream(
        OSError("stop the loop after adoption"), release_event
    )
    pa_retry.open.return_value = retry_stream

    recorder = _make_recorder(module, pa_retry, init_probe=pa1)
    # 1: __init__'s cached construction only -- no attempt has run yet.
    assert module.pyaudio.PyAudio.call_count == 1

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None

    # Bounded wait for adoption: recorder.pyaudio only becomes pa_retry
    # once the retry's open() has succeeded and been adopted.
    assert _wait_until(lambda: recorder.pyaudio is pa_retry)

    # Exactly one fresh construction (retry attempt 2): 1 (__init__) + 1.
    assert module.pyaudio.PyAudio.call_count == 2
    # Cached attempt 1 resolved at __init__ (device -1); the retry
    # re-resolves against its fresh instance (device 7). The rate query
    # (issue #55) makes a second call on the same instance.
    assert pa_retry.get_default_input_device_info.call_count >= 1
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

    recorder = _make_recorder(module, pa_retry, init_probe=pa1)
    # Fast-failure path: both attempts fail immediately (the retry's
    # device lookup raises RuntimeError), so the worker can complete and
    # clear recording before any capture; wait for completion on
    # observable state instead (lens review HIGH #1/#4 read-before-start
    # race).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

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

    recorder = _make_recorder(module, init_probe=pa1)
    recorder._generation = 1
    recorder.recording = False  # session already ended via stop_recording()
    recorder.last_error = "microphone busy \u2014 recording did not start"

    recorder._recording_worker(1)

    assert recorder.last_error == "microphone busy \u2014 recording did not start"
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()


def test_first_attempt_uses_cached_pyaudio_no_construction(
    audio_recorder_module,
):
    """Issue #42 (inverted from
    test_first_attempt_constructs_fresh_pyaudio_and_resolves_device):
    attempt 1 reuses the cached self.pyaudio and its cached device
    index -- no fresh construction and no re-resolution on the happy
    path. self.pyaudio is the __init__'s persistent instance from the
    moment of construction, not None."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    # 1: __init__'s cached construction only -- and self.pyaudio is it.
    assert module.pyaudio.PyAudio.call_count == 1
    assert recorder.pyaudio is pa1
    assert recorder.input_device_index == 0

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread.is_alive()  # worker is blocked in read()

    # No attempt-1 construction: the count is still just the __init__
    # one, asserted while the worker is deterministically alive.
    assert module.pyaudio.PyAudio.call_count == 1
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

    # pa1 is the __init__'s cached instance; attempt 1 reuses it (issue #42).
    recorder = _make_recorder(module, init_probe=pa1)
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

    recorder = _make_recorder(module, pa_retry, init_probe=pa1)
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

    # Both attempts' instances fail -- pin max_attempts to match, so the
    # loop doesn't try a 3rd/4th construction the mock has no instance
    # left to return.
    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)
    assert recorder.start_recording() is True
    # The worker is deterministically alive (blocked in open() on the
    # test-owned event), so no adoption wait is needed before the
    # capture.
    assert _wait_until(lambda: recorder.recording_thread is not None)
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

    recorder = _make_recorder(module, init_probe=pa1)

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

    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)

    def fake_sleep(_seconds):
        # Simulate the hotkey being released while the worker is asleep
        # between attempts.
        recorder.recording = False

    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    # Fast-abort pattern: fake_sleep flips the state synchronously during
    # the sleep, so the worker may tear down before the thread handle is
    # captured; wait for completion on observable state instead (lens
    # review HIGH #1/#4).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # The abort must land before the retry attempt does any work: 1
    # cached (__init__) + 0 (no retry construction) = 1, no open() call
    # on the retry instance.
    assert module.pyaudio.PyAudio.call_count == 1
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

    recorder = _make_recorder(module, init_probe=pa1)
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

    recorder = _make_recorder(module, init_probe=pa1)
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
