import math
import queue
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
import pyaudio

from kuiskaus.audio_retry import (
    MAX_ATTEMPTS,
    PA_INTERNAL_ERROR_ERRNO,
    RETRY_BACKOFF_SECONDS,
    attempt_open_once,
    format_microphone_error,
    log_retry_attempt,
    refresh_pyaudio_session,
    terminate_quietly,
)

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
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        max_attempts: int = MAX_ATTEMPTS,
        retry_backoff_seconds: Sequence[float] = RETRY_BACKOFF_SECONDS,
        on_capture_started: Callable[[], None] | None = None,
    ):
        # Defensive state first, before validation can raise: __del__
        # -> cleanup() can run on a partially-constructed instance without
        # a hasattr guard (issue #37 lens review MEDIUM #5).
        self.pyaudio: pyaudio.PyAudio | None = None
        self.stream: pyaudio.Stream | None = None
        self.recording = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.recording_thread: threading.Thread | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._generation = 0
        # Monotonic start timestamp of the current/last recording
        # (issue #40), used for the [audio.stop] duration_ms field.
        self._start_monotonic: float | None = None
        self.on_capture_started = on_capture_started
        self._capture_announced = False
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if max_attempts > 1 and not retry_backoff_seconds:
            raise ValueError(
                "retry_backoff_seconds must be non-empty when "
                "max_attempts > 1 (pass max_attempts=1 for no retries)"
            )
        if any(not math.isfinite(s) or s < 0 for s in retry_backoff_seconds):
            raise ValueError(
                "retry_backoff_seconds must contain only finite, non-negative values"
            )
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self._stuck_open_timeout_seconds: float = sum(retry_backoff_seconds) + 0.5
        # Persistent cached PyAudio session (issue #42).
        try:
            self.pyaudio = pyaudio.PyAudio()
        except Exception as construct_error:  # noqa: BLE001 - PyAudio()
            # construction wraps PortAudio's Pa_Initialize(), whose
            # failure modes aren't documented as a narrow exception set.
            self.pyaudio = None
            print(
                "Warning: PyAudio construction failed at startup "
                f"({construct_error}); will retry at first recording."
            )
        self.input_device_index: int | None = None
        if self.pyaudio is not None:
            try:
                self.input_device_index = self._find_default_input_device(self.pyaudio)
            except (OSError, RuntimeError):
                # Transient (coreaudiod storm at startup) or no device at
                # all -- the worker's poll + the retry loop re-resolve.
                self.input_device_index = None
                print(
                    "Warning: microphone probe failed at startup; "
                    "will re-resolve at first recording."
                )

    @property
    def current_generation(self) -> int:
        """Public read-only view of the recording generation counter
        (issue #40/43 lens review MEDIUM #2): UI code (menubar drain
        gate, capture-started enqueue) reads the current generation
        through this accessor, not the private _generation attribute.
        Read lock-free: int reads are atomic under the GIL and every
        mutation happens under _lock."""
        return self._generation

    def _find_default_input_device(self, pa: "pyaudio.PyAudio") -> int:
        """Find the default system microphone for the given PyAudio instance."""
        try:
            info = pa.get_default_input_device_info()
            return info["index"]
        except (OSError, KeyError, TypeError):
            # Fallback to first available input device
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    return i
            raise RuntimeError("No input device found")

    @staticmethod
    def _close_stream_quietly(stream: "pyaudio.Stream") -> None:
        """Best-effort stream teardown; a close failure must not propagate."""
        try:
            stream.stop_stream()
            stream.close()
        except OSError as e:
            print(f"Error closing stream: {e}")

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
        my_gen: int,
        attempt: int,
        attempt_start: float,
    ) -> bool:
        """Lock-scoped local-capture ownership transfer: adopt new_pa and
        its stream into self.pyaudio, terminating the previous session's
        instance. A retry (attempt >= 2) adopts a fresh instance and
        disposes the previous; attempt 1 (issue #42) adopts the cached
        instance. Returns False if superseded while retrying."""
        with self._lock:
            if self._generation != my_gen or not self.recording:
                self._close_stream_quietly(stream)
                if attempt > 1:
                    # A fresh (unadopted) retry instance: this worker
                    # owns its termination -- cleanup() will never see
                    # it (self.pyaudio was never reassigned).
                    terminate_quietly(new_pa)
                log_retry_attempt(
                    attempt, self.max_attempts, attempt_start, None, "abort"
                )
                return False
            # Capture the previous instance locally and reassign under
            # the same lock acquisition that writes self.pyaudio, so a
            # concurrent cleanup() can never observe a torn state.
            old_pyaudio = self.pyaudio
            self.pyaudio = new_pa
            # Attempt 1 reused the cached instance: it remains current
            # and its device index is still valid.
            if attempt == 1:
                return True

        # Outside the lock: whoever captures old_pyaudio owns its
        # termination; cleanup() uses the same pattern, so
        # double-terminate is impossible by construction.
        #
        # Invariant (lens review HIGH #3): self.stream is ALWAYS None
        # here -- only _recording_worker writes it, AFTER
        # _open_stream_with_retry returns, so the post-lock read is a
        # defensive guard, not a racy gate. No leak on any current path.
        if old_pyaudio is not None and self.stream is None:
            terminate_quietly(old_pyaudio)
        return True

    def _open_stream_with_retry(self, my_gen: int) -> "pyaudio.Stream | None":
        """Open the input stream, retrying with backoff up to
        ``self.max_attempts``. Attempt 1 reuses the cached ``self.pyaudio``
        and ``self.input_device_index`` (issue #42); retries 2..N construct
        a fresh PyAudio() and re-resolve the device (issue #37, unchanged).
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

            pa, stream, error = attempt_open_once(
                pyaudio,
                self.format,
                self.channels,
                self.sample_rate,
                self.chunk_size,
                self._find_default_input_device,
                self.max_attempts,
                attempt,
                attempt_start,
                # Issue #42: attempt 1 reuses the cached session; its
                # device was already resolved against the cache at
                # construction/poll time (issue #38's per-attempt
                # re-resolution stays on retries 2..N).
                existing_pa=self.pyaudio if attempt == 1 else None,
                existing_device_index=self.input_device_index if attempt == 1 else None,
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
                pa, stream, my_gen, attempt, attempt_start
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
        """Device-change poll + on-demand construction (issue #42),
        called at the top of _recording_worker before the retry loop.
        Delegates to audio_retry.refresh_pyaudio_session (extracted to
        keep this module under the 500-line limit).
        """
        with self._lock:
            self.pyaudio, self.input_device_index = refresh_pyaudio_session(
                pyaudio,
                self.pyaudio,
                self.input_device_index,
                self._find_default_input_device,
            )

    def _recording_worker(self, my_gen: int) -> None:
        """Worker thread for continuous audio recording.

        Every shared-state write here (failure path, adoption,
        post-loop teardown) is gated on ``self._generation == my_gen``
        so a late/stale worker can never clobber a newer recording's
        state. The device-change poll (issue #42) runs first: it
        rebuilds the cached PyAudio session when the default input
        device moved, and constructs on-demand if __init__ failed.
        """
        self._refresh_pyaudio_session()
        stream = self._open_stream_with_retry(my_gen)
        if stream is None:
            return  # last_error already set (or generation superseded)
        with self._lock:
            if self._generation != my_gen:
                self._close_stream_quietly(stream)
                return
            self.stream = stream
            # Only clear last_error if this generation's session is
            # still active. If self.recording is already False here, the
            # session was ended (e.g. stop_recording()'s stuck-open path
            # already surfaced an error for this generation) before this
            # late open() returned; clobbering that error would hide it
            # from a release handler that may have already read it. The
            # while loop below still breaks immediately (recording is
            # False).
            if self.recording:
                self.last_error = None

        while True:
            with self._lock:
                if self._generation != my_gen or not self.recording:
                    break
                # Announce capture-start at most once per start_recording()
                # cycle (issue #43 task-c). The predicate is deliberately
                # not re-checked here: the generation/recording gate below
                # (re-acquired before the callback fires) is the actual
                # staleness defence, and the flag is reset once per cycle.
                capture_announced = self._capture_announced
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
            except OSError as e:
                # CoreAudio/pyaudio I/O failure: log and stop the recording loop
                print(f"Error recording audio: {e}")
                break
            self.audio_queue.put(data)
            if not capture_announced and len(data) > 0:
                with self._lock:
                    if (
                        self._capture_announced
                        or self._generation != my_gen
                        or not self.recording
                    ):
                        continue
                    self._capture_announced = True
                callback = self.on_capture_started
                if callback is not None:
                    try:
                        callback()
                    except Exception:  # noqa: BLE001 - callback boundary
                        # A raising callback must not stop the read loop;
                        # the recording continues.
                        print("on_capture_started callback raised")
                        continue

        self._close_stream_quietly(stream)

        with self._lock:
            if self._generation == my_gen:
                self.recording = False
                self.stream = None
                self.recording_thread = None

    def start_recording(self) -> bool:
        """Start recording audio.

        Returns True if a new recording was admitted and a worker
        spawned, False if refused because a worker is already alive.
        Liveness (not ``self.recording``) is the sole refusal signal:
        after stop_recording()'s stuck-open path, self.recording is
        already False while recording_thread may still be blocked in a
        native pyaudio call. Gating refusal on self.recording as well
        would let a second worker call pyaudio.open() on the same
        shared self.pyaudio concurrently with the still-running one --
        exactly the native-level hazard issue #16 closes. Liveness
        alone is race-free: once a thread is observed not-alive it can
        never become alive again, so the reassignment below is always
        safe.
        """
        with self._lock:
            if self.recording_thread is not None and self.recording_thread.is_alive():
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
            # Clear before spawning: clearing after thread.start() could
            # wipe an error the new worker has already set.
            self.last_error = None
            self.audio_queue = queue.Queue()  # Clear any old data
            self.recording_thread = threading.Thread(
                target=self._recording_worker, args=(my_gen,), daemon=True
            )
            thread = self.recording_thread

        thread.start()
        return True

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the audio data as numpy array.

        Every empty-array return logs exactly one structured line
        ``[audio.stop] chunks=<n> duration_ms=<m> reason=<value>``
        (issue #40). The reason is classified from observable state:
        no-worker (no live session, no error), retry-exhausted (no
        live session, last_error set), stuck-open (worker still alive
        after the join timeout), or race-lost (clean join, empty
        queue, no error).
        """
        with self._lock:
            was_recording = self.recording
            thread = self.recording_thread
            my_gen = self._generation
            start_monotonic = self._start_monotonic
            if was_recording:
                self.recording = False
                self._start_monotonic = None

        if not was_recording:
            reason = "no-worker" if self.last_error is None else "retry-exhausted"
            print(f"[audio.stop] chunks=0 duration_ms=0 reason={reason}")
            return np.array([], dtype=np.float32)

        stuck_open = False
        if thread is not None:
            thread.join(timeout=self._stuck_open_timeout_seconds)
            if thread.is_alive():
                # Stuck-open detection: without this, a stuck (as opposed
                # to failed) open surfaces as "no speech" because
                # last_error hasn't been written yet.
                with self._lock:
                    if self._generation == my_gen:
                        self.last_error = "microphone busy — recording did not start"
                print(
                    "Recording worker still alive after stop; microphone may be stuck"
                )
                stuck_open = True

        # Collect all audio data
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())

        if audio_chunks:
            # Convert to numpy array
            audio_data = b"".join(audio_chunks)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            return audio_float

        if start_monotonic is not None:
            duration_ms = int((time.monotonic() - start_monotonic) * 1000)
        else:
            duration_ms = 0
        reason = "stuck-open" if stuck_open else "race-lost"
        print(
            f"[audio.stop] chunks={len(audio_chunks)} "
            f"duration_ms={duration_ms} reason={reason}"
        )
        return np.array([], dtype=np.float32)

    def cleanup(self) -> None:
        """Clean up PyAudio resources.

        terminate() is skipped -- and logged -- if a worker may still
        be alive, since terminate() while another thread holds/uses a
        stream is unsafe. Same lock-scoped local-capture pattern as the
        between-recordings adoption (issue #37 task-c): whoever captures
        self.pyaudio into a local owns its termination, so a concurrent
        adoption and cleanup() can never double-terminate.
        """
        if self.recording:
            self.stop_recording()

        with self._lock:
            if self.recording or (
                self.recording_thread is not None and self.recording_thread.is_alive()
            ):
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
        """Ensure cleanup on deletion"""
        self.cleanup()
