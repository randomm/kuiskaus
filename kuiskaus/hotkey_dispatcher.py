"""Ordered single-threaded dispatcher for hotkey callbacks.

Feed events to `dispatch` from any thread (e.g. the macOS event-tap
callback); they run serially on a dedicated worker in submission order,
so a release always observes the state left by the press it belongs to.
"""

import queue
import threading
from collections.abc import Callable

_SENTINEL: object = object()


class HotkeyDispatcher:
    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def dispatch(self, callback: Callable[[], None]) -> None:
        """Submit `callback` for ordered execution on the worker thread."""
        self._queue.put(callback)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            item()

    def shutdown(self) -> None:
        """Stop the worker (best effort; does not drain the queue)."""
        if self._thread.is_alive():
            self._queue.put(_SENTINEL)
            self._thread.join(timeout=2.0)
