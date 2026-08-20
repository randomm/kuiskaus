"""Single-threaded executor for hotkey press/release callbacks (issue #17).

Both hotkey listeners (``hotkey_listener.py`` and
``hotkey_listener_cgevent.py``) run their user callbacks through one
instance of :class:`CallbackDispatcher` each. The event handler enqueues
the callback; a single worker thread drains the queue in enqueue order.
This guarantees that a release callback always runs after the press
callback that started it, so the release handler observes the state its
press set — without ever blocking the run loop that delivers events.
"""

import queue
import threading
import time
from collections.abc import Callable

# None is the stop sentinel: it can never be a real callback (dispatch()
# only accepts Callable[[], None]), so `is not None` narrows the queue
# item back to Callable for the type checker.


class CallbackDispatcher:
    """Queue + one worker thread: callbacks run serially in event order."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def dispatch(self, callback: Callable[[], None]) -> None:
        """Queue a callback; it runs on the worker, in enqueue order."""
        self._queue.put(callback)

    def start(self) -> None:
        """Start the worker thread (no-op if already running)."""
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker after any pending callbacks finish.

        If a callback is still running past ``timeout`` seconds, the
        worker thread is left alive and ``_worker`` keeps referencing it,
        so a later start() refuses to spawn a second worker draining the
        same queue -- two workers racing on the same queue would
        reintroduce the ordering bug this dispatcher exists to fix.
        """
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            print(
                "CallbackDispatcher: worker did not stop within "
                f"{timeout}s; refusing to start a second worker until it exits"
            )
            return
        self._worker = None

    def drain(self, timeout: float = 5.0) -> None:
        """Block until the queue has been fully processed (tests).

        Bounded by ``timeout`` seconds. ``queue.Queue.join()`` has no
        timeout parameter, so an unbounded ``drain()`` would hang the
        calling thread forever if a future regression ever let a
        callback die without its matching ``task_done()`` -- exactly
        the failure class this dispatcher exists to fix (issue #17
        review finding). Raising loudly here turns that into a fast,
        visible test failure instead of a silent CI hang.
        """
        deadline = time.monotonic() + timeout
        poll_interval = 0.01
        while self._queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"CallbackDispatcher.drain() timed out after {timeout}s "
                    "with callbacks still unfinished"
                )
            time.sleep(poll_interval)

    def _worker_loop(self) -> None:
        while True:
            callback = self._queue.get()
            if callback is None:
                # Stop sentinel: unblock queue.join() callers, then exit
                # the loop so stop() doesn't leave a zombie worker behind.
                self._queue.task_done()
                return
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - worker boundary: one
                # callback's failure must not kill the worker thread or
                # leave every later callback silently swallowed until
                # the app restarts (issue #17 review finding).
                print(f"CallbackDispatcher: callback raised {exc!r}")
            finally:
                self._queue.task_done()
