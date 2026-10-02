"""Shared helpers for the test_audio_recorder test modules (issue #70).

Hardware-free unit tests for AudioRecorder (issue #16).

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

import re
import threading
import time
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np


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


def _make_pyaudio_instance(
    index: int = 0,
    rate: float | None = None,
) -> MagicMock:
    """A mock PyAudio() instance with a resolvable default input device.

    ``rate`` adds a native ``defaultSampleRate`` to the device info (issue #55).
    """
    device_info: dict = {"index": index}
    if rate is not None:
        device_info["defaultSampleRate"] = rate
    instance = MagicMock(name=f"PyAudioInstance-{index}")
    instance.get_default_input_device_info.return_value = device_info
    # Issue #55 lens review MEDIUM (security): the open-time native-rate
    # query now targets the resolved device via get_device_info_by_index
    # (the default-device info is only used to resolve the index), so the
    # fixture mirrors the rate on that path too.
    instance.get_device_info_by_index.return_value = device_info
    return instance


def _make_recorder(
    module,
    *pa_instances: MagicMock | Exception,
    init_probe: MagicMock | None = None,
    extra_pa: MagicMock | None = None,
    **kwargs,
):
    """Build an AudioRecorder wired for the cached-instance shape (issue
    #42: PR #38's per-recording construction is retired).

    ``module.pyaudio.PyAudio`` yields, in order: ``init_probe`` (the
    __init__'s persistent cached instance -- synthesized when not given),
    then ``pa_instances`` as successive fresh constructions (retry-loop
    attempts 2..N and device-change poll rebuilds), then ``extra_pa``
    (synthesized when not given) on every further construction (a later
    recording cycle's poll, etc.).

    An ``Exception`` instance in ``pa_instances`` is raised instead of
    returned, simulating a PyAudio() construction failure on that slot.
    """
    if init_probe is None:
        init_probe = _make_pyaudio_instance(-1)
    if extra_pa is None:
        extra_pa = _make_pyaudio_instance(999)

    queue: list = [init_probe, *pa_instances]

    def _next_pa(*_args, **_kwargs) -> MagicMock:
        item = queue.pop(0) if queue else extra_pa
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]  # queue holds MagicMocks

    module.pyaudio.PyAudio = MagicMock(side_effect=_next_pa)
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


def _capture_callback_recorder(
    module, callback, read_data: bytes, **recorder_kwargs
) -> tuple:
    """Build a recorder wired with `callback` whose worker adopts a stream
    whose read() returns read_data once (firing the capture-started hook)
    and then blocks on a test-owned release event before raising OSError
    to end the loop. Returns (recorder, thread, release_event).

    ``recorder_kwargs`` (e.g. ``extra_pa=`` for a second recording cycle
    on the same recorder) are forwarded to ``_make_recorder``."""
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    stream = MagicMock(name="stream")
    stream.read.side_effect = _blocking_read_then_block(read_data, release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(
        module, init_probe=pa1, on_capture_started=callback, **recorder_kwargs
    )
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


def _drain_worker_queue(recorder) -> bytes:
    """Drain the worker's audio queue without deadlocking on its read loop.

    While the worker is blocked in read() (a test-owned release event
    governs its next read), the queue contents are stable; the test can
    drain them here and replicate the stop_recording() normalize +
    resample to verify the 16 kHz contract without needing the worker
    to join (setting the release event would let its next read yield
    more frames and defeat the drain). The fixture teardown
    (_cleanup_recorder) reclaims the daemon worker.
    """
    drained = b""
    while not recorder.audio_queue.empty():
        drained += recorder.audio_queue.get()
    return drained


def _streaming_stream(release: threading.Event) -> MagicMock:
    """A stream whose read() returns an endless supply of 1024-frame int16
    chunks (paced), until ``release`` is set."""
    chunk = (np.ones(1024, dtype=np.int16) * 1000).tobytes()

    def fake_read(*_args, **_kwargs):
        if release.wait(timeout=0.002):
            raise OSError("stream closed by test")
        return chunk

    stream = MagicMock(name="warm-stream")
    stream.read.side_effect = fake_read
    return stream


def _warm_recorder(module, stream):
    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.return_value = stream
    recorder = _make_recorder(module, init_probe=pa)
    recorder.set_keep_warm(True)
    return recorder, pa
