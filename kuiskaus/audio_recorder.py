import queue
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
import pyaudio

from kuiskaus.audio_device import PyAudioRegistry, check_superseded
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
    open_warm_stream,
    refresh_pyaudio_session,
    validate_retry_config,
    worker_alive,
)
from kuiskaus.audio_warm import WarmCapture

# Aliases for annotations written inside the class body, where the
# ``pyaudio`` property (issue #72) otherwise shadows the module name.
_PyAudio = pyaudio.PyAudio
_Stream = pyaudio.Stream

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
        # The device session exists before anything else: __del__ ->
        # cleanup() runs on a partially-constructed instance, and the
        # pyaudio attributes resolve through it.
        self._session = PyAudioRegistry(pyaudio)
        # Defensive state next: cleanup() reads it without a hasattr guard.
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
        # Persistent cached session (issue #42), tracked by the device
        # session's registry so terminate_all() can drive PortAudio's init
        # count to zero for a device refresh (issue #72).
        self._session.open(find_default_input_device)

    @property
    def pyaudio(self) -> "_PyAudio | None":
        """The live cached session (issue #42), owned by the device
        session's registry so terminate_all() reaches every instance."""
        return self._session.pyaudio

    @pyaudio.setter
    def pyaudio(self, pa: "_PyAudio | None") -> None:
        self._session.pyaudio = pa

    @property
    def input_device_index(self) -> int | None:
        """Device index resolved against ``pyaudio`` (issue #42)."""
        return self._session.input_device_index

    @input_device_index.setter
    def input_device_index(self, index: int | None) -> None:
        self._session.input_device_index = index

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

    def _check_superseded(self, my_gen: int, attempt: int, attempt_start: float) -> bool:  # fmt: skip
        """True (and logged) if this retry sequence lost its race."""
        return check_superseded(
            self._lock,
            lambda: self._generation != my_gen or not self.recording,
            attempt, self.max_attempts, attempt_start,
        )  # fmt: skip

    def _adopt_and_dispose_previous(self, new_pa, stream, capture_rate, my_gen, attempt, attempt_start) -> bool:  # fmt: skip
        """Lock-scoped ownership transfer into the device session, which
        owns the live instance (issue #72). False = superseded."""
        return self._session.adopt(
            self._lock,
            lambda: self._generation != my_gen or not self.recording,
            lambda rate: setattr(self, "_capture_rate", rate),
            lambda: self.stream is not None,
            new_pa, stream, capture_rate, attempt, attempt_start, self.max_attempts,
        )  # fmt: skip

    def _open_stream_with_retry(self, my_gen: int) -> "_Stream | None":
        """Open the input stream, retrying with backoff up to
        ``self.max_attempts``. Every attempt reuses the single live
        ``self.pyaudio`` and the index resolved against it (issue #72):
        constructing one per attempt kept PortAudio's init count above
        zero, so no retry ever re-enumerated devices. Returns the opened
        stream, or None if all attempts failed or the generation was
        superseded."""
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
                # Issue #72: reuse the live instance and the index resolved
                # against it (a None index re-resolves on that same
                # instance) -- never a second live instance alongside one.
                existing_pa=self.pyaudio,
                existing_device_index=self.input_device_index,
                registry=self._session,
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

    def _refresh_pyaudio_session(self, blocked: threading.Thread | None = None) -> None:
        """Live-default device reset (issue #72) at the top of the worker.
        Native calls run OUTSIDE _lock (issue #60): they can block and
        stop_recording() needs the lock. One worker runs this at a time;
        ``blocked`` is the worker just joined -- still alive means busy.
        """
        with self._lock:
            pa, device_index = self.pyaudio, self.input_device_index
        pa, device_index = refresh_pyaudio_session(
            pyaudio, pa, device_index, find_default_input_device, blocked
        )
        with self._lock:
            self.pyaudio, self.input_device_index = pa, device_index

    def _rearm_warm(self, usable: bool) -> None:
        """Re-arm the keep-warm capture after a reset replaced its session.
        usable=False is the busy path: a capture that would not stop is
        dead (its stop event is set, so it can never re-open) -- drop it
        and later presses take the worker path, not a dead sink.
        """
        self._warm = None
        if usable:
            self.set_keep_warm(True)

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

    def _open_warm_stream(self) -> tuple["_Stream", int]:
        """One open for WarmCapture's thread. Never overlaps a per-press
        worker's native open on the shared PyAudio (issue #16)."""
        previous = self.recording_thread
        if previous is not None and previous.is_alive():
            previous.join(timeout=PREVIOUS_WORKER_WAIT_SECONDS)
        self._refresh_pyaudio_session(previous)
        with self._lock:
            pa, device_index = self.pyaudio, self.input_device_index
        new_pa, stream, rate = open_warm_stream(
            pyaudio, pa, device_index, self.format, self.channels,
            self.sample_rate, self.chunk_size, self._session,
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
        self._refresh_pyaudio_session(previous)
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
        worker gets the orphan and joins it before opening; the session
        gate runs a keep-warm press's live-default check (#72).
        """
        if not self._session.press_check_if_admissible(
            self.recording, self._warm, self.recording_thread,
            find_default_input_device, self._rearm_warm,
        ):  # fmt: skip
            self.last_error = "microphone busy — recording did not start"
            return False

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
            self.pyaudio = None
            self.input_device_index = None

        # Issue #72: every registered instance goes -- a single instance
        # anywhere keeps PortAudio's init count above zero and frozen.
        self._session.terminate_all()

    def __del__(self):
        self.cleanup()
