import queue
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
import pyaudio

from kuiskaus.audio_resample import TARGET_RATE, assemble_audio
from kuiskaus.audio_retry import (
    MAX_ATTEMPTS,
    PA_INTERNAL_ERROR_ERRNO,
    PREVIOUS_WORKER_WAIT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    attempt_open_once,
    close_stream_quietly,
    find_default_input_device,
    format_microphone_error,
    log_retry_attempt,
    log_stop,
    open_cached_session,
    open_warm_stream,
    refresh_pyaudio_session,
    terminate_quietly,
    validate_retry_config,
    worker_alive,
)
from kuiskaus.audio_warm import WarmCapture

# Re-exported so `import kuiskaus.audio_recorder` keeps exposing the
# retry-policy constants (tests and docs reference them here).
__all__ = [
    "MAX_ATTEMPTS",
    "PA_INTERNAL_ERROR_ERRNO",
    "RETRY_BACKOFF_SECONDS",
    "AudioRecorder",
]


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = TARGET_RATE,
        chunk_size: int = 1024,
        channels: int = 1,
        max_attempts: int = MAX_ATTEMPTS,
        retry_backoff_seconds: Sequence[float] = RETRY_BACKOFF_SECONDS,
        on_capture_started: Callable[[], None] | None = None,
    ) -> None:
        # Defensive state first: __del__ -> cleanup() can run on a
        # partially-constructed instance without a hasattr guard.
        self.pyaudio: pyaudio.PyAudio | None = None
        self.stream: pyaudio.Stream | None = None
        self.recording = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.recording_thread: threading.Thread | None = None
        self._warm: WarmCapture | None = None  # issue #66
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._capture_rate: int | None = None  # issue #55 (adoption)
        self._start_monotonic: float | None = None  # issue #40
        self.on_capture_started = on_capture_started
        self._capture_announced = False
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        validate_retry_config(max_attempts, retry_backoff_seconds)
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self._stuck_open_timeout_seconds: float = sum(retry_backoff_seconds) + 0.5
        # Persistent cached PyAudio session (issue #42).
        self.pyaudio, self.input_device_index = open_cached_session(pyaudio)

    @property
    def current_generation(self) -> int:
        """Public read-only view of the recording generation counter
        (issue #40/43 lens review MEDIUM #2): UI code (menubar drain
        gate, capture-started enqueue) reads the current generation
        through this accessor, not the private _generation attribute.
        Read lock-free: int reads are atomic under the GIL and every
        mutation happens under _lock."""
        return self._generation

    @property
    def capture_rate(self) -> int | None:
        """Native rate the stream was opened at (issue #55), or None
        before any open succeeds."""
        return self._capture_rate

    def _check_superseded(
        self, my_gen: int, attempt: int, attempt_start: float
    ) -> bool:
        """Return True (and log the abort) if this retry sequence has
        been superseded: a newer generation took over, or recording
        was released. Shared by the pre-sleep and post-sleep abort
        sites (issue #37 lens review MEDIUM #7: previously duplicated)."""
        with self._lock:
            if self._generation != my_gen or not self.recording:
                log_retry_attempt(
                    attempt, self.max_attempts, attempt_start, None, "abort"
                )
                return True
            return False

    def _adopt_and_dispose_previous(
        self,
        new_pa: "pyaudio.PyAudio",
        stream: "pyaudio.Stream",
        capture_rate: int | None,
        my_gen: int,
        attempt: int,
        attempt_start: float,
    ) -> bool:
        """Lock-scoped ownership transfer: adopt new_pa, its stream, and
        the capture rate into self.pyaudio, terminating the previous
        session's instance. A retry (attempt >= 2) adopts a fresh instance
        and disposes the previous; attempt 1 (issue #42) adopts the cached
        instance. Returns False if superseded while retrying."""
        with self._lock:
            if self._generation != my_gen or not self.recording:
                close_stream_quietly(stream)
                # A fresh (unadopted) retry instance: this worker owns its
                # termination -- cleanup() never sees it (not reassigned).
                if attempt > 1:
                    terminate_quietly(new_pa)
                log_retry_attempt(
                    attempt, self.max_attempts, attempt_start, None, "abort"
                )
                return False
            # Capture the previous instance locally and reassign under
            # the same lock acquisition that writes self.pyaudio
            # (issue #55: the capture rate is written alongside the
            # stream so stop_recording() sees an atomic rate + stream).
            old_pyaudio = self.pyaudio
            self.pyaudio = new_pa
            self._capture_rate = capture_rate
            if attempt == 1:
                return True  # cached instance: device index still valid

        # Outside the lock: whoever captures old_pyaudio owns its
        # termination (cleanup() uses the same pattern, so double-
        # terminate is impossible by construction).
        #
        # Invariant (lens review HIGH #3): self.stream is ALWAYS None
        # here -- only _recording_worker writes it, AFTER
        # _open_stream_with_retry returns, so the post-lock read is a
        # defensive guard, not a racy gate.
        if old_pyaudio is not None and self.stream is None:
            terminate_quietly(old_pyaudio)
        return True

    def _open_stream_with_retry(self, my_gen: int) -> "pyaudio.Stream | None":
        """Open the input stream, retrying with backoff up to
        ``self.max_attempts``. Attempt 1 reuses the cached ``self.pyaudio``
        and ``self.input_device_index`` (issue #42); retries 2..N
        construct a fresh PyAudio() and re-resolve the device (issue #37).
        Returns the opened stream, or None if all attempts failed or the
        generation was superseded mid-retry."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            attempt_start = time.monotonic()

            if attempt > 1:
                if self._check_superseded(my_gen, attempt, attempt_start):
                    return None

                log_retry_attempt(
                    attempt, self.max_attempts, attempt_start, None, "sleep"
                )
                # Clamp+reuse: if max_attempts - 1 exceeds
                # len(retry_backoff_seconds), the final backoff value
                # repeats for extra attempts.
                backoff_index = min(attempt - 2, len(self.retry_backoff_seconds) - 1)
                time.sleep(self.retry_backoff_seconds[backoff_index])
                if self._check_superseded(my_gen, attempt, attempt_start):
                    return None

            pa, stream, error, capture_rate = attempt_open_once(
                pyaudio,
                self.format,
                self.channels,
                self.sample_rate,
                self.chunk_size,
                find_default_input_device,
                self.max_attempts,
                attempt,
                attempt_start,
                # Issue #42: attempt 1 reuses the cached session; its
                # device was already resolved against the cache at
                # construction/poll time (issue #38's per-attempt
                # re-resolution stays on retries 2..N).
                existing_pa=self.pyaudio if attempt == 1 else None,
                existing_device_index=(
                    self.input_device_index if attempt == 1 else None
                ),
            )
            if stream is None:
                last_error = error
                if isinstance(error, RuntimeError):
                    break
                continue
            if pa is None:
                raise RuntimeError(
                    "internal invariant violated: attempt_open_once returned a "
                    "stream without a PyAudio instance"
                )
            if not self._adopt_and_dispose_previous(
                pa, stream, capture_rate, my_gen, attempt, attempt_start
            ):
                return None
            log_retry_attempt(attempt, self.max_attempts, attempt_start, None, "adopt")
            return stream

        with self._lock:
            if self._generation == my_gen:
                error_text = (
                    format_microphone_error(last_error)
                    if last_error is not None
                    else "Microphone unavailable: unknown error"
                )
                self.last_error = error_text
                self.recording = False
                self.stream = None
                self.recording_thread = None
        return None

    def _refresh_pyaudio_session(self) -> None:
        """Device-change poll + on-demand construction (issue #42), run at
        the top of _recording_worker; delegates to audio_retry.

        Native calls run OUTSIDE _lock (issue #60): they can block for
        seconds and stop_recording() needs the lock. One worker runs this
        at a time (a new worker waits for its predecessor).
        """
        with self._lock:
            pa, device_index = self.pyaudio, self.input_device_index
        pa, device_index = refresh_pyaudio_session(
            pyaudio, pa, device_index, find_default_input_device
        )
        with self._lock:
            self.pyaudio, self.input_device_index = pa, device_index

    def _announce_capture(self, my_gen: int) -> None:
        """Fire on_capture_started at most once per start_recording() cycle
        (issue #43); the generation/recording gate is the staleness defence."""
        with self._lock:
            if (
                self._capture_announced
                or self._generation != my_gen
                or not self.recording
            ):
                return
            self._capture_announced = True
        callback = self.on_capture_started
        if callback is not None:
            try:
                callback()
            except Exception:  # noqa: BLE001 - callback boundary
                # A raising callback must not stop the capture thread.
                print("on_capture_started callback raised")

    def set_keep_warm(self, enabled: bool) -> None:
        """Keep the input stream open between recordings (issue #66)."""
        if enabled and self._warm is None:
            self._warm = WarmCapture(self._open_warm_stream, self.chunk_size)
            self._warm.start()
        elif not enabled and self._warm is not None:
            warm, self._warm = self._warm, None
            self._capture_rate = warm.capture_rate  # for an in-flight recording
            warm.stop()

    def _open_warm_stream(self) -> tuple["pyaudio.Stream", int]:
        """One open for WarmCapture's thread. Never overlaps a per-press
        worker's native open on the shared PyAudio (issue #16)."""
        previous = self.recording_thread
        if previous is not None and previous.is_alive():
            previous.join(timeout=PREVIOUS_WORKER_WAIT_SECONDS)
        self._refresh_pyaudio_session()
        with self._lock:
            pa, device_index = self.pyaudio, self.input_device_index
        new_pa, stream, rate = open_warm_stream(
            pyaudio, pa, device_index, self.format, self.channels,
            self.sample_rate, self.chunk_size,
        )  # fmt: skip
        if new_pa is not pa:
            with self._lock:
                self.pyaudio = new_pa
        return stream, rate

    def _recording_worker(
        self, my_gen: int, previous: "threading.Thread | None" = None
    ) -> None:
        """Worker thread for continuous audio recording.

        Every shared-state write here (failure path, adoption,
        post-loop teardown) is gated on ``self._generation == my_gen``
        so a late/stale worker can never clobber a newer recording's
        state. The device-change poll (issue #42) runs first: it
        rebuilds the cached PyAudio session when the default input
        device moved, and constructs on-demand if __init__ failed.

        ``previous`` is an orphaned worker still blocked in a native open
        (issue #60): wait for it first so two opens never overlap on the
        shared PyAudio instance.
        """
        if previous is not None:
            previous.join(timeout=PREVIOUS_WORKER_WAIT_SECONDS)
            with self._lock:
                if self._generation != my_gen or not self.recording:
                    return  # released again while waiting
        self._refresh_pyaudio_session()
        stream = self._open_stream_with_retry(my_gen)
        if stream is None:
            return  # last_error already set (or generation superseded)
        with self._lock:
            if self._generation != my_gen:
                close_stream_quietly(stream)
                return
            self.stream = stream
            # Only clear last_error if this generation's session is
            # still active: after stop_recording()'s stuck-open path,
            # self.recording is False while a late open() may still be
            # returning; clobbering last_error there would hide the
            # error a release handler may have already read.
            if self.recording:
                self.last_error = None

        while True:
            with self._lock:
                if self._generation != my_gen or not self.recording:
                    break
                # Announce capture-start at most once per start_recording()
                # cycle (issue #43 task-c); the flag is reset once per
                # cycle and the generation/recording gate below (re-
                # acquired before the callback fires) is the staleness
                # defence.
                capture_announced = self._capture_announced
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
            except OSError as e:
                # CoreAudio/pyaudio I/O failure: log and stop the recording loop
                print(f"Error recording audio: {e}")
                break
            self.audio_queue.put(data)
            if not capture_announced and len(data) > 0:
                self._announce_capture(my_gen)

        close_stream_quietly(stream)

        with self._lock:
            if self._generation == my_gen:
                self.recording = False
                self.stream = None
                self.recording_thread = None

    def start_recording(self) -> bool:
        """Admit a new recording and spawn its worker (issue #16).

        A press during a live recording is refused. A worker orphaned by
        stop_recording()'s stuck-open path (recording=False, still in a
        native open) must not wedge later presses (issue #60), yet two
        opens must not overlap on the shared PyAudio (issue #16): the new
        worker gets the orphan and joins it before opening.
        """
        with self._lock:
            warm = self._warm
            previous = self.recording_thread
            if not worker_alive(previous):
                previous = None
            if self.recording and (warm is not None or previous is not None):
                return False

            if self.recording:
                # Stale state: recording was left True with no live worker
                # (e.g. a worker died in open() without teardown). Recover
                # instead of wedging every future start.
                print("Recovering from stale recording state")
                self.recording = False
                self.stream = None
                self.recording_thread = None

            self._generation += 1
            my_gen = self._generation
            self._start_monotonic = time.monotonic()
            self.recording = True
            self._capture_announced = False
            # Reset alongside the stream/queue state (issue #55).
            self._capture_rate = None
            # Clear before spawning: clearing after thread.start() could
            # wipe an error the new worker has already set.
            self.last_error = None
            self.audio_queue = queue.Queue()  # Clear any old data
            sink = self.audio_queue
            if warm is None:
                self.recording_thread = threading.Thread(
                    target=self._recording_worker,
                    args=(my_gen, previous),
                    daemon=True,
                )
                thread = self.recording_thread

        if warm is not None:
            # Keep-warm (issue #66): no open, no worker; route the ring.
            warm.begin(sink, lambda: self._announce_capture(my_gen))
        else:
            thread.start()
        return True

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the audio as a numpy array.

        Every empty-array return logs one ``[audio.stop] chunks=<n>
        duration_ms=<m> reason=<value>`` line (issue #40): the
        reason is no-worker, retry-exhausted, stuck-open, or
        race-lost (classified from observable state)."""
        with self._lock:
            was_recording = self.recording
            thread = self.recording_thread
            my_gen = self._generation
            start_monotonic = self._start_monotonic
            warm = self._warm
            capture_rate = warm.capture_rate if warm else self._capture_rate
            if was_recording:
                self.recording = False
                self._start_monotonic = None

        if not was_recording:
            reason = "no-worker" if self.last_error is None else "retry-exhausted"
            log_stop(0, None, reason)
            return np.array([], dtype=np.float32)

        stuck_open = False
        if thread is not None:
            thread.join(timeout=self._stuck_open_timeout_seconds)
            if thread.is_alive():
                # Stuck-open detection: a stuck (not failed) open
                # surfaces as "no speech" because last_error hasn't
                # been written yet -- the join timeout is the only
                # observable signal.
                with self._lock:
                    if self._generation == my_gen:
                        self.last_error = "microphone busy — recording did not start"
                print(
                    "Recording worker still alive after stop; microphone may be stuck"
                )
                stuck_open = True

        if warm is not None:
            warm.end()
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())

        if audio_chunks:
            return assemble_audio(audio_chunks, capture_rate)

        reason = "stuck-open" if stuck_open else "race-lost"
        if warm is not None:
            # The warm stream was not delivering yet: report it (#66).
            with self._lock:
                if self._generation == my_gen:
                    self.last_error = "microphone busy — recording did not start"
            reason = "warm-not-ready"
        log_stop(0, start_monotonic, reason)
        return np.array([], dtype=np.float32)

    def cleanup(self) -> None:
        """Clean up PyAudio resources. terminate() is skipped -- and
        logged -- if a worker may still be alive, since terminate() while
        another thread holds a stream is unsafe (lock-scoped local-capture
        pattern, issue #37 task-c).
        """
        if self.recording:
            self.stop_recording()

        warm, self._warm = self._warm, None
        if warm is not None and not warm.stop():
            print("Warm capture still active at cleanup; skipping PyAudio.terminate()")
            return

        worker_thread = self.recording_thread
        with self._lock:
            if self.recording or worker_alive(worker_thread):
                print(
                    "Recording worker still active at cleanup; skipping "
                    "PyAudio.terminate()"
                )
                return
            old_pyaudio = self.pyaudio
            self.pyaudio = None

        if old_pyaudio is not None:
            terminate_quietly(old_pyaudio)

    def __del__(self):
        self.cleanup()
