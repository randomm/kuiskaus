"""Keep-mic-warm capture with a rolling pre-roll buffer (issue #66).

pa.open can block for seconds inside CoreAudio (issue #60), and speech
spoken before it returns is lost when the stream is opened per key press.
WarmCapture keeps one input stream open on a background thread. Idle audio
is held only in a short in-memory ring (never stored); begin() hands the
ring plus every later chunk to a recording queue, end() stops routing.
"""

import queue
import threading
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

from kuiskaus.audio_retry import close_stream_quietly

if TYPE_CHECKING:
    import pyaudio

PREROLL_SECONDS = 0.5
REOPEN_DELAY_SECONDS = 2.0


class WarmCapture:
    def __init__(
        self,
        open_stream: "Callable[[], tuple[pyaudio.Stream, int]]",
        chunk_size: int,
        preroll_seconds: float = PREROLL_SECONDS,
        reopen_delay: float = REOPEN_DELAY_SECONDS,
    ) -> None:
        self._open_stream = open_stream
        self._chunk_size = chunk_size
        self._preroll_seconds = preroll_seconds
        self._reopen_delay = reopen_delay
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ring: deque[bytes] = deque()
        self._sink: queue.Queue[bytes] | None = None
        self._on_first_chunk: Callable[[], None] | None = None
        self._ready = False
        self._rate: int | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def capture_rate(self) -> int | None:
        return self._rate

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop the thread. False if it is still alive (stuck in a native
        open/read), in which case the caller must not terminate PortAudio."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def begin(
        self,
        sink: "queue.Queue[bytes]",
        on_first_chunk: Callable[[], None] | None = None,
    ) -> None:
        """Seed ``sink`` with the pre-roll and route every later chunk to
        it. ``on_first_chunk`` fires once, on the first live chunk."""
        with self._lock:
            for chunk in self._ring:
                sink.put(chunk)
            self._ring.clear()
            self._sink = sink
            self._on_first_chunk = on_first_chunk

    def end(self) -> None:
        with self._lock:
            self._sink = None
            self._on_first_chunk = None

    def _deliver(self, data: bytes) -> None:
        with self._lock:
            if self._sink is None:
                self._ring.append(data)
                return
            self._sink.put(data)
            announce, self._on_first_chunk = self._on_first_chunk, None
        if announce is not None:
            announce()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                stream, rate = self._open_stream()
            except Exception as e:  # noqa: BLE001 - worker thread boundary, logged
                print(f"Warm capture open failed: {e}")
                self._stop.wait(self._reopen_delay)
                continue
            with self._lock:
                self._rate = rate
                self._ring = deque(
                    maxlen=max(1, int(self._preroll_seconds * rate / self._chunk_size))
                )
                self._ready = True
            try:
                while not self._stop.is_set():
                    data = stream.read(self._chunk_size, exception_on_overflow=False)
                    self._deliver(data)
            except OSError as e:
                print(f"Warm capture read failed: {e}")
            finally:
                self._ready = False
                close_stream_quietly(stream)
            self._stop.wait(self._reopen_delay)
