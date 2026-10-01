"""Hardware-free tests for WarmCapture (issue #66).

A fake stream stands in for PyAudio: every read() returns one numbered
chunk after a short wait, so tests can assert ordering and pre-roll
without any real device. All waits are bounded.
"""

import queue
import threading
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from kuiskaus.audio_warm import WarmCapture

CHUNK = 4  # samples per chunk -> 8 bytes of int16
RATE = 400  # 0.5 s pre-roll at this rate / CHUNK -> 50 chunks


def _wait_until(predicate: Callable[[], object], timeout: float = 3.0) -> bool:
    deadline = threading.Event()
    for _ in range(int(timeout / 0.005)):
        if predicate():
            return True
        deadline.wait(0.005)
    return bool(predicate())


class FakeStream:
    """Yields chunk n as bytes([n % 256, 0]) * CHUNK, one per read."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.count = 0
        self.fail_after = fail_after
        self.closed = threading.Event()
        self._pace = threading.Event()

    def read(self, _n: int, exception_on_overflow: bool = False) -> bytes:
        self._pace.wait(0.002)
        if self.fail_after is not None and self.count >= self.fail_after:
            raise OSError("device gone")
        self.count += 1
        return bytes([self.count % 256, 0]) * CHUNK

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed.set()


@pytest.fixture
def warm():
    made: list[WarmCapture] = []

    def build(open_stream, **kwargs) -> WarmCapture:
        capture = WarmCapture(open_stream, CHUNK, reopen_delay=0.01, **kwargs)
        made.append(capture)
        return capture

    yield build
    for capture in made:
        capture.stop()


def test_becomes_ready_and_reports_capture_rate(warm):
    stream = FakeStream()
    capture = warm(lambda: (stream, RATE))
    assert capture.ready is False
    capture.start()
    assert _wait_until(lambda: capture.ready)
    assert capture.capture_rate == RATE


def test_begin_seeds_preroll_then_streams_new_chunks(warm):
    capture = warm(lambda: (FakeStream(), RATE))
    capture.start()
    assert _wait_until(lambda: capture.ready)
    assert _wait_until(lambda: len(capture._ring) >= 3)

    sink: queue.Queue[bytes] = queue.Queue()
    capture.begin(sink)
    seeded = sink.qsize()
    assert seeded >= 3
    assert _wait_until(lambda: sink.qsize() > seeded)  # live chunks follow

    chunks = [sink.get_nowait() for _ in range(sink.qsize())]
    firsts = [c[0] for c in chunks]
    assert firsts == sorted(firsts)  # in order, no gaps or duplicates
    assert firsts == list(range(firsts[0], firsts[0] + len(firsts)))


def test_end_stops_routing_back_to_ring(warm):
    capture = warm(lambda: (FakeStream(), RATE))
    capture.start()
    assert _wait_until(lambda: capture.ready)
    sink: queue.Queue[bytes] = queue.Queue()
    capture.begin(sink)
    assert _wait_until(lambda: sink.qsize() > 2)
    capture.end()
    settled = sink.qsize()
    threading.Event().wait(0.1)
    assert sink.qsize() == settled


def test_preroll_is_bounded(warm):
    capture = warm(lambda: (FakeStream(), RATE), preroll_seconds=0.5)
    capture.start()
    assert _wait_until(lambda: capture.ready)
    assert _wait_until(lambda: capture._ring.maxlen is not None)
    maxlen = capture._ring.maxlen
    assert maxlen == int(0.5 * RATE / CHUNK)
    assert _wait_until(lambda: len(capture._ring) == maxlen)
    threading.Event().wait(0.1)
    assert len(capture._ring) == maxlen


def test_first_chunk_callback_fires_once_per_begin(warm):
    capture = warm(lambda: (FakeStream(), RATE))
    capture.start()
    assert _wait_until(lambda: capture.ready)
    announced = MagicMock()
    sink: queue.Queue[bytes] = queue.Queue()
    capture.begin(sink, on_first_chunk=announced)
    assert _wait_until(lambda: announced.call_count >= 1)
    threading.Event().wait(0.05)
    assert announced.call_count == 1


def test_failed_open_is_retried(warm):
    attempts = []

    def flaky_open():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("coreaudio stall")
        return FakeStream(), RATE

    capture = warm(flaky_open)
    capture.start()
    assert _wait_until(lambda: capture.ready)
    assert len(attempts) == 3


def test_read_error_reopens_stream(warm):
    streams = [FakeStream(fail_after=2), FakeStream()]

    def open_stream():
        return streams.pop(0), RATE

    capture = warm(open_stream)
    capture.start()
    assert _wait_until(lambda: not streams)  # second stream was opened
    assert _wait_until(lambda: capture.ready)


def test_stop_ends_thread_and_closes_stream(warm):
    stream = FakeStream()
    capture = warm(lambda: (stream, RATE))
    capture.start()
    assert _wait_until(lambda: capture.ready)
    assert capture.stop() is True
    assert stream.closed.is_set()
    assert capture.ready is False


def test_stop_reports_false_when_open_is_stuck(warm):
    release = threading.Event()

    def stuck_open():
        release.wait(10.0)
        raise OSError("late")

    capture = warm(stuck_open)
    capture.start()
    assert capture.stop(timeout=0.1) is False
    release.set()
