"""Unit tests for CallbackDispatcher (issue #17).

The single-threaded executor is the highest-risk piece of the fix: if
``stop()`` fails to terminate the worker, every listener quit path hangs
and, on restart, a second worker can end up draining the same queue,
reintroducing the exact ordering race issue #17 exists to eliminate.
These tests assert on that behaviour directly rather than "does not
raise".
"""

import threading
import time

from kuiskaus.callback_dispatcher import CallbackDispatcher


class TestCallbackDispatcherStop:
    def test_stop_terminates_worker_thread(self):
        """After stop(), the worker thread must actually be dead."""
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        worker = dispatcher._worker
        assert worker is not None
        dispatcher.stop()
        worker.join(timeout=1.0)
        assert not worker.is_alive()

    def test_stop_returns_promptly(self):
        """stop() must not block for its 5s join timeout on the happy path."""
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        start = time.monotonic()
        dispatcher.stop()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_stop_is_idempotent(self):
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        dispatcher.stop()
        dispatcher.stop()  # must not raise

    def test_stop_does_not_spawn_a_second_worker_when_callback_outlives_timeout(self):
        """If a callback is still running when stop()'s timeout elapses,
        the worker must be left alive and referenced, so a later start()
        refuses to spawn a second worker draining the same queue."""
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        started = threading.Event()
        release_slow = threading.Event()

        def slow():
            started.set()
            release_slow.wait(timeout=2.0)

        dispatcher.dispatch(slow)
        assert started.wait(timeout=1.0)
        zombie_worker = dispatcher._worker
        assert zombie_worker is not None

        dispatcher.stop(timeout=0.05)
        # Timeout elapsed while the callback was still running: stop()
        # must not have cleared _worker, and the thread is still alive.
        assert dispatcher._worker is zombie_worker
        assert zombie_worker.is_alive()

        dispatcher.start()
        # start() must refuse to spawn a second worker against the same
        # queue while the first one is still draining it.
        assert dispatcher._worker is zombie_worker

        release_slow.set()
        zombie_worker.join(timeout=2.0)
        assert not zombie_worker.is_alive()
        # cleanup: real stop() would do this once alive is False
        dispatcher._worker = None

    def test_restart_does_not_leak_a_second_worker(self):
        """A stopped-then-restarted dispatcher must have exactly one live
        worker draining the queue, not two racing on the same queue."""
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        first_worker = dispatcher._worker
        assert first_worker is not None
        dispatcher.stop()
        assert not first_worker.is_alive()

        dispatcher.start()
        second_worker = dispatcher._worker
        assert second_worker is not None
        try:
            assert second_worker is not first_worker
            assert second_worker.is_alive()

            events: list[tuple[str, int]] = []
            for i in range(5):
                dispatcher.dispatch(lambda i=i: events.append(("run", i)))
            dispatcher.drain()
            assert events == [("run", i) for i in range(5)]
        finally:
            dispatcher.stop()


class TestCallbackDispatcherOrdering:
    def test_callbacks_run_in_enqueue_order_on_one_thread(self):
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        events: list[str] = []
        try:
            dispatcher.dispatch(lambda: events.append("a"))
            dispatcher.dispatch(lambda: events.append("b"))
            dispatcher.dispatch(lambda: events.append("c"))
            dispatcher.drain()
        finally:
            dispatcher.stop()
        assert events == ["a", "b", "c"]

    def test_slow_callback_blocks_later_callbacks_until_done(self):
        """Single-threaded executor: a later callback cannot start before
        an earlier one finishes, even if the earlier one is slow."""
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        events: list[str] = []
        release_slow = threading.Event()

        def slow():
            events.append("slow-start")
            release_slow.wait(timeout=2.0)
            events.append("slow-end")

        try:
            dispatcher.dispatch(slow)
            dispatcher.dispatch(lambda: events.append("fast"))
            time.sleep(0.05)
            # "fast" must not have run yet: it's queued behind "slow".
            assert events == ["slow-start"]
            release_slow.set()
            dispatcher.drain()
        finally:
            dispatcher.stop()
        assert events == ["slow-start", "slow-end", "fast"]


class TestCallbackDispatcherFailureIsolation:
    def test_raising_callback_does_not_kill_the_worker(self):
        dispatcher = CallbackDispatcher()
        dispatcher.start()
        events: list[str] = []
        try:
            dispatcher.dispatch(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            dispatcher.dispatch(lambda: events.append("still-runs"))
            dispatcher.drain()
        finally:
            dispatcher.stop()
        assert events == ["still-runs"]
