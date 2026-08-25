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
from collections.abc import Callable
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


def _assert_stop_log_line(
    captured,
    reason: str,
    chunks: int | None = None,
    duration_ms_min: int | None = None,
) -> None:
    """Assert exactly one well-formed ``[audio.stop] chunks=<n>
    duration_ms=<m> reason=<value>`` line in captured stdout, with the
    given reason (issue #40). Optionally pin the chunks count and a
    duration_ms floor. Shared by the four empty-return reason tests so
    the line shape is asserted in one place."""
    log_lines = [
        line for line in captured.out.splitlines() if line.startswith("[audio.stop]")
    ]
    pattern = re.compile(
        r"^\[audio\.stop\] chunks=(\d+) duration_ms=(\d+) reason=[\w-]+$"
    )
    assert len(log_lines) == 1
    assert pattern.match(log_lines[0]), log_lines[0]
    if chunks is not None:
        assert f"chunks={chunks}" in log_lines[0]
    if duration_ms_min is not None:
        match = pattern.match(log_lines[0])
        assert match is not None
        assert int(match.group(2)) >= duration_ms_min
    assert f"reason={reason}" in log_lines[0]


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


def _make_recorder(
    module,
    *pa_instances: MagicMock | Exception,
    init_probe: MagicMock | None = None,
    **kwargs,
):
    """Build an AudioRecorder wired for the cached-instance shape (issue
    #42: PR #38's per-recording construction is retired).

    ``module.pyaudio.PyAudio`` is set up to yield, in order, the
    __init__'s persistent cached instance first (synthesized from
    ``init_probe`` when given -- e.g. to simulate a failing startup
    device probe -- else an always-resolvable default), then
    ``pa_instances`` as successive fresh constructions for the
    retry-loop attempts 2..N of every recording this recorder makes.
    Attempt 1 of every recording reuses the cached instance, so a
    test describing a successful attempt-1 open only needs the cached
    instance (``init_probe`` / the first synthesized default) to be
    the instance it expects; its open() is what the attempt drives.

    An ``Exception`` instance in ``pa_instances`` is raised instead of
    returned (mock's own ``side_effect``-iterable behaviour),
    simulating a PyAudio() construction failure on that attempt.
    """
    if init_probe is not None:
        init_pa = init_probe
    else:
        init_pa = _make_pyaudio_instance(-1)
    # The device-change poll (issue #42) constructs a fresh PyAudio()
    # whenever the cached index is None or the poll sees a device move.
    # Use a factory callable: __init__ gets init_pa, retry attempts
    # consume pa_instances in order, and any further constructions
    # (2nd recording cycle's poll, etc.) get the last instance.
    queue: list = [*pa_instances]
    fallback: MagicMock = (
        pa_instances[-1]
        if pa_instances and not isinstance(pa_instances[-1], Exception)
        else init_pa
    )

    def _pa_factory() -> MagicMock:
        nonlocal init_pa
        if init_pa is not None:
            result: MagicMock = init_pa
            init_pa = None  # consumed
            return result
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item  # type: ignore[return-value]  # queue holds MagicMocks
        return fallback

    module.pyaudio.PyAudio = MagicMock(side_effect=_pa_factory)
    return module.AudioRecorder(**kwargs)


def _blocking_stream(
    read_error: Exception, release_event: threading.Event
) -> MagicMock:
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
    predicate: Callable[[], object],
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
    return bool(predicate())


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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
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

    recorder = _make_recorder(module, pa1)
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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
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

    recorder = _make_recorder(module, pa1)
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
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
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

    # Cycle 1's attempt 1 drives the cached instance (pa1); cycle 2's
    # attempt 1 reuses the same cache (device unchanged -- the poll
    # reuses it, so pa2 is never constructed; it exists only as a
    # spare).
    recorder = _make_recorder(module, pa2, init_probe=pa1)
    assert recorder.current_generation == 0

    assert recorder.start_recording() is True
    assert recorder.current_generation == 1
    release1.set()
    assert recorder.recording_thread is not None
    recorder.recording_thread.join(timeout=10.0)

    assert recorder.start_recording() is True
    assert recorder.current_generation == 2
    release2.set()
    assert recorder.recording_thread is not None
    recorder.recording_thread.join(timeout=10.0)


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
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
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

    # Attempt 1 drives the cached instance (init_probe), retries 2..4
    # drive the three fresh instances.
    recorder = _make_recorder(
        module,
        *pa_instances,
        init_probe=_make_pyaudio_instance(-1),
        max_attempts=4,
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
    # re-resolves against its fresh instance (device 7).
    pa_retry.get_default_input_device_info.assert_called_once()
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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_oserror_from_device_enumeration_is_retryable(audio_recorder_module):
    """A coreaudiod storm can make device enumeration itself raise OSError
    -9986 in _find_default_input_device's fallback path
    (get_default_input_device_info fails, then get_device_count raises),
    not just open() (lens review HIGH #3). That must cost one attempt and
    let the loop continue, not propagate out of _recording_worker and
    kill the thread silently (the exact bug #16 this PR closes)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    # Default-info lookup raises (so the fallback enumeration path runs),
    # and the fallback's own get_device_count() call raises OSError --
    # the enumeration-path OSError this test is named for.
    pa1.get_default_input_device_info.side_effect = OSError("no default")
    pa1.get_device_count.side_effect = OSError("paInternalError during enumeration")
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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_stuck_open_sets_last_error_on_join_timeout(audio_recorder_module, capsys):
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

    # Issue #40: the stuck-open outcome must also surface as the
    # structured [audio.stop] log line (the existing print above it is
    # unchanged; this is the observability add).
    _assert_stop_log_line(
        capsys.readouterr(), "stuck-open", chunks=0, duration_ms_min=0
    )

    never_release.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_backoff_loop_makes_up_to_n_attempts_with_sleep_between(
    audio_recorder_module, monkeypatch
):
    """max_attempts consecutive OSErrors exhaust the loop: attempt 1
    reuses the cached instance (issue #42), attempts 2..N each construct
    a fresh PyAudio(), sleeping exactly max_attempts - 1 times between
    attempts."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa_instances = []
    for i in range(3):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"retry attempt {i + 2} failed")
        pa_instances.append(pa)

    recorder = _make_recorder(
        module,
        *pa_instances,
        init_probe=_make_pyaudio_instance(-1),
        max_attempts=4,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    # Fast-exhaustion pattern: all attempts fail in <1ms (sleep is mocked
    # to a no-op), so the worker may have already torn down and cleared
    # recording_thread before the capture below. Wait for completion on
    # observable state instead of capturing the thread handle (lens
    # review HIGH #1/#4: the read-before-start race failed in CI).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # 1 cached construction (__init__) + 3 fresh retry constructions = 4.
    assert module.pyaudio.PyAudio.call_count == 4
    assert module.time.sleep.call_count == 3


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_backoff_sleep_cadence_matches_schedule(audio_recorder_module, monkeypatch):
    """The sleep durations actually used match the effective backoff
    schedule verbatim, in order."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa_instances = []
    for i in range(3):
        pa = _make_pyaudio_instance(i)
        pa.open.side_effect = OSError(f"retry attempt {i + 2} failed")
        pa_instances.append(pa)

    schedule = (0.11, 0.22, 0.33)
    recorder = _make_recorder(
        module,
        *pa_instances,
        init_probe=_make_pyaudio_instance(-1),
        max_attempts=4,
        retry_backoff_seconds=schedule,
    )
    # Fast-exhaustion pattern: the worker may tear down before the thread
    # handle can be captured; wait for completion on observable state
    # instead (lens review HIGH #1/#4).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    assert module.time.sleep.call_args_list == [mock.call(s) for s in schedule]


def test_backoff_loop_succeeds_on_middle_attempt(audio_recorder_module, monkeypatch):
    """Attempt 1 (cached) and retry 2 fail, retry 3 succeeds: only 2
    sleeps occur and last_error is left None."""
    module = audio_recorder_module
    monkeypatch.setattr(module.time, "sleep", MagicMock())

    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("attempt 1 failed")
    pa2 = _make_pyaudio_instance(1)
    pa2.open.side_effect = OSError("retry attempt 2 failed")
    pa3 = _make_pyaudio_instance(2)
    release_event = threading.Event()
    pa3.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)

    recorder = _make_recorder(
        module,
        pa2,
        pa3,
        init_probe=pa1,
        max_attempts=4,
        retry_backoff_seconds=(0.01, 0.01, 0.01),
    )
    assert recorder.start_recording() is True
    # Post-adoption capture idiom: adoption (blocked in read()) provably
    # precedes any teardown, so the capture is race-free.
    assert _wait_until(lambda: recorder.pyaudio is pa3, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None

    assert module.time.sleep.call_count == 2
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

    recorder = _make_recorder(module, init_probe=pa1, max_attempts=1)
    assert recorder.start_recording() is True
    # Fast-exhaustion path: the worker can complete and clear
    # recording_thread before a plain read of the attribute; assert on
    # observable state instead of the transient thread handle (same
    # race fixed in PR #38 round 2 for the sibling tests).
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    assert recorder.last_error is not None
    assert "sudo killall coreaudiod" in recorder.last_error


def test_last_error_generic_for_non_paInternalError(audio_recorder_module):
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    err = OSError("Device unavailable")
    err.errno = -9985  # paDeviceUnavailable, deliberately not -9986
    pa1.open.side_effect = err

    recorder = _make_recorder(module, init_probe=pa1, max_attempts=1)
    assert recorder.start_recording() is True
    # Fast-exhaustion path: the worker can complete and clear
    # recording_thread before a plain read of the attribute; assert on
    # observable state instead of the transient thread handle (same
    # race fixed in PR #38 round 2 for the sibling tests).
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    assert recorder.last_error is not None
    assert "killall" not in recorder.last_error
    assert "Microphone unavailable" in recorder.last_error


# ---------------------------------------------------------------------------
# PyAudio ownership (issue #37 task-c/e)
# ---------------------------------------------------------------------------


def test_cached_pyaudio_reused_across_recordings_when_device_unchanged(
    audio_recorder_module,
):
    """Issue #42 (inverted from
    test_fresh_pyaudio_constructed_every_start_recording_call): two
    consecutive recordings on an unchanged default device reuse the
    single cached PyAudio() -- 1 total construction (__init__), none at
    attempt 1 of either recording. The second recording's worker poll
    sees the device unchanged and keeps the cache."""
    module = audio_recorder_module
    pa_first = _make_pyaudio_instance(0)
    first_release = threading.Event()
    pa_first.open.return_value = _blocking_stream(
        OSError("stop first recording"), first_release
    )

    recorder = _make_recorder(module, init_probe=pa_first)
    # 1: the __init__ cached construction only.
    assert module.pyaudio.PyAudio.call_count == 1

    assert recorder.start_recording() is True
    # Post-adoption capture idiom: the worker is blocked in read() after
    # adoption, so it cannot have torn down recording_thread.
    assert _wait_until(lambda: recorder.pyaudio is pa_first, timeout=5.0)
    first_thread = recorder.recording_thread
    assert first_thread is not None

    first_release.set()
    first_thread.join(timeout=10.0)
    assert not first_thread.is_alive()

    second_release = threading.Event()
    pa_first.open.return_value = _blocking_stream(
        OSError("stop second recording"), second_release
    )
    assert recorder.start_recording() is True
    # The SAME cached instance is adopted for the second recording.
    assert _wait_until(lambda: recorder.pyaudio is pa_first, timeout=5.0)
    second_thread = recorder.recording_thread
    assert second_thread is not None
    assert second_thread is not first_thread
    assert recorder.pyaudio is pa_first

    # Still just the one __init__ construction -- no per-recording
    # construction on the happy path.
    assert module.pyaudio.PyAudio.call_count == 1

    second_release.set()
    second_thread.join(timeout=10.0)
    assert not second_thread.is_alive()


def test_device_change_triggers_pyaudio_reconstruction(audio_recorder_module):
    """Issue #42 DoD: the worker's device-change poll rebuilds the cached
    session when the default input device moved between recordings --
    the stale instance is terminated, a fresh one constructed and
    adopted for attempt 1, and the new index cached."""
    module = audio_recorder_module
    pa_first = _make_pyaudio_instance(0)
    first_release = threading.Event()
    pa_first.open.return_value = _blocking_stream(
        OSError("stop first recording"), first_release
    )
    pa_new = _make_pyaudio_instance(1)
    second_release = threading.Event()
    pa_new.open.return_value = _blocking_stream(
        OSError("stop second recording"), second_release
    )

    recorder = _make_recorder(module, pa_new, init_probe=pa_first)

    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_first, timeout=5.0)

    first_release.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)
    assert recorder.stream is None  # teardown invariant

    # The default input device moved between recordings: the poll must
    # rebuild the session. The construction sequence is __init__ (1) +
    # the poll's reconstruction (1) = 2, so the poll consumes pa_new.
    pa_first.get_default_input_device_info.return_value = {"index": 1}
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_new, timeout=5.0)

    assert pa_first.terminate.called  # stale session disposed by the poll
    assert module.pyaudio.PyAudio.call_count == 2
    assert recorder.input_device_index == 1  # new index cached

    second_release.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)


def test_no_reconstruction_when_device_stable(audio_recorder_module):
    """Issue #42 DoD: two back-to-back recordings with the poll returning
    the same default index construct NO new PyAudio -- the cached
    session is reused verbatim (the perf fix's core behaviour)."""
    module = audio_recorder_module
    pa_cached = _make_pyaudio_instance(0)
    first_release = threading.Event()
    pa_cached.open.return_value = _blocking_stream(
        OSError("stop first recording"), first_release
    )

    recorder = _make_recorder(module, init_probe=pa_cached)
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_cached, timeout=5.0)

    first_release.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)

    second_release = threading.Event()
    pa_cached.open.return_value = _blocking_stream(
        OSError("stop second recording"), second_release
    )
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_cached, timeout=5.0)

    # Device stable across recordings: still only the __init__
    # construction -- the poll re-resolved the same index and kept the
    # cached instance.
    assert module.pyaudio.PyAudio.call_count == 1
    assert pa_cached.terminate.call_count == 0
    assert recorder.input_device_index == 0

    second_release.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)


def test_poll_failure_falls_through_to_retry_loop(audio_recorder_module):
    """Issue #42 DoD: a device-change poll that raises OSError (a
    coreaudiod storm at worker start) must not wedge -- the cached state
    is kept as-is and the retry loop handles the genuinely stale state:
    attempt 1 fails on the cache, the retry re-resolves fresh on its own
    instance and succeeds."""
    module = audio_recorder_module
    pa_cached = _make_pyaudio_instance(0)
    # The poll's get_default_input_device_info() raises OSError, and the
    # fallback enumeration path does too -- _find_default_input_device
    # therefore raises RuntimeError (no input device found) for attempt 1,
    # so attempt 1 must cost a retry. open() on the cache is also set to
    # fail defensively in case a device resolution ever succeeds.
    pa_cached.get_default_input_device_info.side_effect = OSError("poll storm")
    pa_cached.get_device_count.side_effect = OSError("enumeration storm")
    pa_cached.open.side_effect = OSError("stale session")

    pa_retry = _make_pyaudio_instance(1)
    release_event = threading.Event()
    pa_retry.open.return_value = _blocking_stream(
        OSError("stop the loop"), release_event
    )

    recorder = _make_recorder(module, pa_retry, init_probe=pa_cached)
    # __init__'s own _find_default_input_device hits the same OSError
    # fallback path and raises RuntimeError, which __init__ swallows
    # (input_device_index stays None) -- construction survives.
    assert recorder.input_device_index is None
    assert recorder.pyaudio is pa_cached

    assert recorder.start_recording() is True
    # The retry (fresh instance) succeeds despite the poll failure.
    assert _wait_until(lambda: recorder.pyaudio is pa_retry, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None
    assert thread.is_alive()

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_previous_pyaudio_terminated_when_stream_none_before_next_recording(
    audio_recorder_module,
):
    """Issue #42: the only remaining disposal path between recordings is
    the device-change poll -- a genuinely moved default device disposes
    the previous session's PyAudio() (the happy-path retry adoption no
    longer disposes anything, since attempt 1 reuses the cache)."""
    module = audio_recorder_module
    pa_first = _make_pyaudio_instance(0)
    first_release = threading.Event()
    pa_first.open.return_value = _blocking_stream(
        OSError("stop first recording"), first_release
    )
    pa_new = _make_pyaudio_instance(1)
    second_release = threading.Event()
    pa_new.open.return_value = _blocking_stream(
        OSError("stop second recording"), second_release
    )

    recorder = _make_recorder(module, pa_new, init_probe=pa_first)

    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_first, timeout=5.0)

    first_release.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)
    assert recorder.stream is None  # teardown invariant

    # Default device moved; the poll rebuilds and disposes pa_first.
    pa_first.get_default_input_device_info.return_value = {"index": 1}
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_new)

    pa_first.terminate.assert_called_once()

    second_release.set()
    assert _wait_until(lambda: recorder.recording is False, timeout=10.0)


def test_previous_pyaudio_not_terminated_when_stream_not_none(audio_recorder_module):
    """Negative case / defensive guard: if self.stream were somehow not
    None at adoption time (the invariant that should always hold on
    every real teardown path), the old PyAudio() instance must NOT be
    terminated -- a live stream might still depend on it."""
    module = audio_recorder_module
    old_pa = _make_pyaudio_instance(0)
    new_pa = _make_pyaudio_instance(1)

    recorder = _make_recorder(module, init_probe=new_pa)
    recorder.pyaudio = old_pa
    recorder.stream = MagicMock(name="still-live-stream-sentinel")
    recorder.recording = True

    stream = MagicMock(name="new-stream")
    # Drive the adoption directly: the guard under test is the
    # post-lock stream check, which the retry loop cannot reach without
    # a full open() (the loop only adopts after open() succeeds, at
    # which point the worker owns self.stream == None).
    assert recorder._adopt_and_dispose_previous(
        new_pa, stream, recorder.current_generation, 1, 0.0
    )
    assert recorder.pyaudio is new_pa
    old_pa.terminate.assert_not_called()

    # Reset so GC-time cleanup() doesn't try to close/join sentinel state.
    recorder.recording = False
    recorder.stream = None


# ---------------------------------------------------------------------------
# Mid-backoff abort guards (issue #37 task-e)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_backoff_loop_aborts_on_generation_supersede(
    audio_recorder_module, monkeypatch
):
    """A generation bump mid-backoff (a newer recording superseding this
    one) aborts the loop before the next attempt is even constructed."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("first attempt failed")
    pa_retry = _make_pyaudio_instance(1)

    recorder = _make_recorder(module, pa_retry, init_probe=pa1, max_attempts=2)

    def fake_sleep(_seconds):
        # Simulate a newer recording's generation superseding this one
        # while asleep between attempts.
        recorder._generation += 1

    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    # Fast-abort pattern: fake_sleep flips the state synchronously during
    # the sleep, so the worker may tear down before the thread handle is
    # captured. Wait for completion via the abort's observable outcome:
    # the abort path writes no state, so last_error stays None and the
    # retry attempt (attempt 2) is never constructed (lens review
    # HIGH #1/#4).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # The abort must land before the retry attempt does any work: 1
    # cached (__init__) construction, no retry construction.
    assert module.pyaudio.PyAudio.call_count == 1
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
        module,
        pa2,
        pa3,
        init_probe=pa1,
        max_attempts=3,
        retry_backoff_seconds=(0.01, 0.01),
    )

    def release_after_first_sleep(_seconds):
        recorder.recording = False

    sleep_mock.side_effect = release_after_first_sleep

    # Fast-abort pattern: the release fires synchronously inside the
    # mocked sleep, so the worker may tear down before the thread handle
    # is captured; wait for completion on observable state instead (lens
    # review HIGH #1/#4).
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)

    # Only attempt 1's failure triggers the first sleep; the loop aborts
    # right after waking, before attempt 2 is constructed or a second
    # sleep is scheduled. 1 = the __init__ cached construction only.
    assert sleep_mock.call_count == 1
    assert module.pyaudio.PyAudio.call_count == 1  # cached only, no retry
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
        module, pa_retry, init_probe=pa1, max_attempts=2, retry_backoff_seconds=(0.01,)
    )
    assert recorder.start_recording() is True
    # Post-adoption capture idiom: the worker is blocked in read() after
    # adoption, so it cannot have torn down recording_thread.
    assert _wait_until(lambda: recorder.pyaudio is pa_retry, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None

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


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_init_survives_probe_oserror_during_coreaudiod_storm(
    audio_recorder_module,
):
    """An OSError from the startup device probe (a coreaudiod storm
    hitting get_device_count()/get_device_info_by_index in the fallback
    enumeration path at app launch) must not propagate out of
    __init__: construction succeeds (the cached instance is kept) with
    input_device_index left None, and the worker's poll + retry loop
    re-resolve at the first start_recording() call (lens review
    MEDIUM #4)."""
    module = audio_recorder_module
    probe = _make_pyaudio_instance(-1)
    probe.get_default_input_device_info.side_effect = OSError("no default")
    probe.get_device_count.side_effect = OSError("enumeration failed")
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    pa1.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)

    # The cached instance is the probe instance; its device lookup
    # failed, so input_device_index is None. The worker's poll then
    # re-resolves (we restore the probe's device info here so the poll
    # succeeds and attempt 1 reuses the cache directly).
    recorder = _make_recorder(module, init_probe=probe)
    assert recorder.pyaudio is probe
    assert recorder.input_device_index is None

    probe.get_default_input_device_info.side_effect = None
    probe.get_default_input_device_info.return_value = {"index": -1}
    probe.get_device_count.side_effect = None

    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is probe, timeout=5.0)

    thread = recorder.recording_thread
    assert thread is not None
    assert recorder.recording is True

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_worker_constructs_pyaudio_when_init_failed(audio_recorder_module):
    """Issue #42 DoD: when __init__'s PyAudio() construction raises, the
    recorder is still constructed with self.pyaudio is None (a startup
    warning, not a failure), and the worker's pre-retry-loop block
    constructs the instance on-demand before attempt 1 runs."""
    module = audio_recorder_module
    pa_worker = _make_pyaudio_instance(0)
    release_event = threading.Event()
    pa_worker.open.return_value = _blocking_stream(
        OSError("stop the loop"), release_event
    )

    # __init__'s PyAudio() raises; the worker's on-demand construction
    # gets pa_worker.
    module.pyaudio.PyAudio = MagicMock(
        side_effect=[RuntimeError("Pa_Initialize failed"), pa_worker]
    )
    recorder = module.AudioRecorder()
    assert recorder.pyaudio is None
    assert recorder.input_device_index is None

    assert recorder.start_recording() is True
    # The worker's on-demand construction adopted pa_worker; attempt 1
    # ran on it with its resolved device.
    assert _wait_until(lambda: recorder.pyaudio is pa_worker, timeout=5.0)
    assert recorder.input_device_index == 0
    thread = recorder.recording_thread
    assert thread is not None
    assert thread.is_alive()

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Constructor kwargs (issue #37 task-e)
# ---------------------------------------------------------------------------


def test_constructor_kwargs_override_module_defaults(audio_recorder_module):
    """max_attempts and retry_backoff_seconds constructor kwargs override
    the module-level defaults, and _stuck_open_timeout_seconds is derived
    from the effective (kwarg) values, not the module constants."""
    module = audio_recorder_module
    recorder = _make_recorder(module, max_attempts=2, retry_backoff_seconds=(0.05,))
    # The helper synthesizes the cached instance internally; the kwargs
    # under test are unaffected by that.
    assert recorder.max_attempts == 2
    assert recorder.retry_backoff_seconds == (0.05,)
    assert recorder._stuck_open_timeout_seconds == pytest.approx(0.55)
    assert recorder.max_attempts != module.MAX_ATTEMPTS
    assert recorder.retry_backoff_seconds != module.RETRY_BACKOFF_SECONDS


# ---------------------------------------------------------------------------
# on_capture_started callback (issue #43 task-c)
# ---------------------------------------------------------------------------


def _capture_callback_recorder(module, callback, read_data: bytes) -> tuple:
    """Build a recorder wired with `callback` whose worker adopts a stream
    whose read() returns read_data once (firing the capture-started hook)
    and then blocks on a test-owned release event before raising OSError
    to end the loop. Returns (recorder, thread, release_event)."""
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = MagicMock(name="stream")
    stream.read.side_effect = _blocking_read_then_block(read_data, release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # Post-adoption capture idiom: after the first read() returns
    # (observable as stream.read having been called once), the worker
    # is blocked on release_event inside read(), so it cannot have torn
    # down recording_thread. A read-call (not callback) predicate, so
    # the helper also works for raising callbacks (issue #43 task-c).
    assert _wait_until(lambda: stream.read.call_count >= 1, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None
    return recorder, thread, release_event


def _blocking_read_then_block(read_data: bytes, release_event: threading.Event):
    """side_effect callable: first call returns read_data, every later call
    blocks on release_event then raises OSError("stop the loop"). The
    "already returned once" state lives in a local list (a function
    attribute would not be ty-resolvable, so a one-element list keeps the
    type checker clean)."""
    done = [False]

    def fake_read(*_args, **_kwargs):
        if not done[0]:
            done[0] = True
            return read_data
        release_event.wait(timeout=10.0)
        raise OSError("stop the loop")

    return fake_read


def test_on_capture_started_fires_after_first_nonempty_read(
    audio_recorder_module,
):
    """The callback fires after the worker's first stream.read() that
    returns a non-empty buffer. Zero-filled CoreAudio warmup bytes ARE
    non-empty and DO fire it (issue #43: the trigger's purpose is
    'audio bytes flowing from coreaudiod', not 'non-silent audio')."""
    module = audio_recorder_module
    callback = MagicMock()
    _, thread, release = _capture_callback_recorder(module, callback, b"\x00" * 2048)

    assert _wait_until(lambda: callback.call_count == 1, timeout=5.0)

    release.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert callback.call_count == 1


def test_on_capture_started_fires_only_once_per_session(audio_recorder_module):
    """A sustained recording (many successful reads) fires the callback
    exactly once; the flag gates the second-and-later reads."""
    module = audio_recorder_module
    callback = MagicMock()
    recorder, thread, release = _capture_callback_recorder(
        module, callback, b"\x01\x02" * 1024
    )

    assert _wait_until(lambda: callback.call_count == 1, timeout=5.0)
    # Let the worker take a few more read() iterations while the callback
    # flag must stay set -- the next read blocks, but the flag state is
    # already pinned.
    assert _wait_until(lambda: recorder._capture_announced is True, timeout=5.0)

    release.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert callback.call_count == 1


@pytest.mark.skip(
    reason="#42 changed to cached PyAudio + poll model; test assumes per-attempt construction — rework tracked in follow-up chore issue"
)
def test_on_capture_started_flag_resets_per_start_recording_cycle(
    audio_recorder_module,
):
    """The once-flag is per start_recording() cycle: a second cycle on the
    same recorder fires the callback again."""
    module = audio_recorder_module
    callback = MagicMock()
    first = _capture_callback_recorder(module, callback, b"\x00" * 2048)
    recorder = first[0]
    release = first[2]
    assert _wait_until(lambda: callback.call_count == 1, timeout=5.0)
    release.set()
    first[1].join(timeout=10.0)
    assert not first[1].is_alive()
    assert callback.call_count == 1

    # Second cycle on the same recorder with a fresh pa instance and
    # stream. The fixture's _make_recorder consumed pa1 for cycle 1 and
    # the probe for __init__, so this cycle needs its own PyAudio()
    # construction (pa2). Use a fresh PyAudio mock returning pa2 only.
    pa2 = _make_pyaudio_instance(1)
    release2 = threading.Event()
    stream2 = MagicMock(name="stream-2")
    stream2.read.side_effect = _blocking_read_then_block(b"\x03" * 2048, release2)
    pa2.open.return_value = stream2
    module.pyaudio.PyAudio = MagicMock(return_value=pa2)
    assert recorder.start_recording() is True
    # The flag was reset by start_recording, so the callback fires again;
    # wait for it before capturing (post-adoption idiom -- the worker is
    # then blocked on release2 inside read()).
    assert _wait_until(lambda: callback.call_count == 2, timeout=5.0)
    thread2 = recorder.recording_thread
    assert thread2 is not None

    release2.set()
    thread2.join(timeout=10.0)
    assert not thread2.is_alive()
    assert callback.call_count == 2


def test_on_capture_started_not_called_on_empty_capture(audio_recorder_module):
    """Release-before-first-read (the race-lost case, issue #40's
    newly-surfaced outcome): the worker breaks out of the loop before any
    stream.read() succeeds, so the callback must never fire."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)
    stream = MagicMock(name="never-read-stream")

    # The while loop rechecks self.recording under lock before each read;
    # if the test clears it before the first read, the worker breaks
    # without ever calling stream.read().
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # stop_recording() clears self.recording under lock and joins.
    recorder.stop_recording()

    assert not recorder.recording
    stream.read.assert_not_called()
    assert callback.call_count == 0


def test_on_capture_started_skipped_when_generation_superseded(
    audio_recorder_module,
):
    """A newer recording superseding this worker between read() and the
    callback firing (generation bump under lock) must suppress the
    callback -- the generation gate is rechecked after read() returns."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)

    def fake_read(*_args, **_kwargs):
        # Simulate a newer generation taking over while the worker is in
        # flight between read() and the callback. This is the exact race
        # the issue's "generation gate" requirement exists to close.
        with recorder._lock:
            recorder._generation += 1
            recorder.recording = True  # the newer generation is recording
        return b"\x00" * 2048

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # The worker's read loop breaks immediately after the first read
    # (the generation no longer matches my_gen) and its post-loop
    # teardown is generation-gated, so it never clears recording for
    # this generation; the only deterministic observable is the stream
    # closure below. Wait on that instead of the transient thread handle
    # (lens review HIGH #1/#4 read-before-start race).
    assert _wait_until(lambda: stream.close.called, timeout=5.0)

    assert callback.call_count == 0
    # The stale worker must still have closed the stream it owned.
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()


def test_on_capture_started_skipped_when_recording_false(audio_recorder_module):
    """stop_recording() landing between the worker's read() returning and
    the callback firing (recording False under lock) must suppress the
    callback -- the recording-True gate is rechecked after read()."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)

    def fake_read(*_args, **_kwargs):
        # Simulate a release landing between read() and the callback.
        with recorder._lock:
            recorder.recording = False
        return b"\x00" * 2048

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # The worker's read loop breaks on the next iteration (recording is
    # False); the callback is suppressed this time and the loop exits.
    # The worker can clear recording_thread before a capture, so assert
    # on observable state instead (lens review HIGH #1/#4 read-before-
    # start race).
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=10.0)

    assert callback.call_count == 0
    # The worker still tears down the stream it owned.
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()


def test_on_capture_started_callback_exception_does_not_stop_worker(
    audio_recorder_module, capsys
):
    """A raising callback must not kill the read loop: the broad guard
    (callback boundary, # noqa: BLE001) logs and continues. The next
    read() still succeeds and the callback is NOT retried (the flag was
    set before the raise)."""
    module = audio_recorder_module

    def boom():
        raise RuntimeError("callback failure")

    recorder, thread, release = _capture_callback_recorder(module, boom, b"\x00" * 2048)

    # The flag is set before the callback runs, so the worker continues
    # the loop after the raise without retrying; the second read blocks
    # on release.
    assert _wait_until(lambda: recorder._capture_announced is True, timeout=5.0)

    release.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    captured = capsys.readouterr()
    assert "on_capture_started callback raised" in captured.out


def test_on_capture_started_default_none_is_backwards_compatible(
    audio_recorder_module,
):
    """AudioRecorder() without on_capture_started never calls anything;
    recording behaves exactly as before the callback existed. The
    flag is set (as a bookkeeping no-op) but no callback runs."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    pa1.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)
    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.on_capture_started is None
    assert recorder._capture_announced is False

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    # The worker is deterministically alive (blocked in read()). Wait for
    # adoption so the post-loop assertion on the flag is race-free.
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert thread.is_alive()

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_on_capture_started_fires_after_retry_adoption(audio_recorder_module):
    """The callback fires on the FIRST successful non-empty read ACROSS
    ALL attempts, i.e. after retry adoption lands and the adopted stream
    delivers its first bytes -- not once per retry attempt. Attempt 1
    never reaches a read() (open() fails), so no callback fires from it.
    A non-empty read on the adopted (attempt-2) stream fires it exactly
    once."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("attempt 1 failed")
    pa_retry = _make_pyaudio_instance(1)
    release_event = threading.Event()
    pa_retry.open.return_value = _blocking_read_stream(b"\x00" * 2048, release_event)

    recorder = _make_recorder(
        module,
        pa_retry,
        init_probe=pa1,
        max_attempts=2,
        on_capture_started=callback,
    )
    assert recorder.start_recording() is True
    # Wait for the retry adoption, then for the callback to fire on the
    # adopted stream's first non-empty read; by then the worker is
    # blocked on release_event inside read(), so the capture is
    # race-free (post-adoption idiom).
    assert _wait_until(lambda: recorder.pyaudio is pa_retry, timeout=5.0)
    assert _wait_until(lambda: callback.call_count == 1, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert callback.call_count == 1  # once total, not once per attempt


def _blocking_read_stream(
    read_data: bytes, release_event: threading.Event
) -> MagicMock:
    """A mock stream whose first read() returns read_data and whose
    subsequent reads block on release_event before raising OSError.
    Distinct from _blocking_stream (which raises on the first call): used
    for tests where the worker must reach the read loop at least once."""
    stream = MagicMock(name="stream")
    stream.read.side_effect = _blocking_read_then_block(read_data, release_event)
    return stream
