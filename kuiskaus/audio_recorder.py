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
    log_retry_attempt,
    log_stop,
    microphone_unavailable,
    validate_retry_config,
    worker_alive,
)
from kuiskaus.audio_warm import WarmCapture
from kuiskaus.audio_warm_arm import (
    WarmArm,
    announce_capture,
    cleanup_recorder_session,
    open_warm_capture_stream,
    refresh_recorder_session,
    warm_press_gate,
)

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
        self.last_error: str | None = None
        self._lock = threading.Lock()
        # The keep-warm capture (issue #66), armed only through WarmArm
        # so the transition is atomic across the #72 caller threads.
        self._warm_arm = WarmArm(self._session.warm, self._lock)
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
        self._session.set_index(index)

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

    @property
    def _warm(self) -> WarmCapture | None:
        """The keep-warm capture slot; it lives in WarmArm, where every
        transition is serialised (issue #72 finding 1)."""
        return self._warm_arm.capture

    def _stale(self, my_gen: int) -> Callable[[], bool]:
        """Supersession predicate for one generation, evaluated by the
        callee under _lock -- its single arbiter (issue #55)."""
        return lambda: self._generation != my_gen or not self.recording

    def _check_superseded(self, my_gen: int, attempt: int, attempt_start: float) -> bool:  # fmt: skip
        """True (and logged) if this retry sequence lost its race."""
        return check_superseded(
            self._lock,
            self._stale(my_gen),
            attempt, self.max_attempts, attempt_start,
        )  # fmt: skip

    def _adopt_and_dispose_previous(self, new_pa, stream, capture_rate, my_gen, attempt, attempt_start) -> bool:  # fmt: skip
        """Lock-scoped ownership transfer into the device session, which
        owns the live instance (issue #72). False = superseded."""
        return self._session.adopt(
            self._lock,
            self._stale(my_gen),
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

            # Issue #72: reuse the live instance and the index resolved
            # against it (a None index re-resolves on that same instance) --
            # never a second live instance alongside one. The pair comes from
            # one snapshot(): an index is only a fact about the instance that
            # enumerated it, so the two must not be read across an instance
            # swap (#72 review, minor 1).
            live_pa, live_device_index = self._session.snapshot()
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
                existing_pa=live_pa,
                existing_device_index=live_device_index,
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
                self.last_error = microphone_unavailable(last_error)
                self.recording = False
                self.stream = None
                self.recording_thread = None
        return None

    def _refresh_pyaudio_session(self, blocked: threading.Thread | None = None) -> None:
        """Live-default device reset (issue #72) at the top of the worker.
        Native calls run OUTSIDE _lock (issue #60): they can block and
        stop_recording() needs the lock. The behaviour is in
        ``audio_warm_arm.refresh_recorder_session``, the seam the warm
        reopen shares; ``blocked`` is the worker this one just joined.
        """
        refresh_recorder_session(self, blocked)

    def _rearm_warm(self, usable: bool) -> None:
        """Re-arm the keep-warm capture after a reset replaced its session.
        usable=False is the busy path: a capture that would not stop is
        dead for presses (its stop event is set, so it can never re-open)
        -- drop it and later presses take the worker path, not a dead sink.
        The session keeps the handle on its live thread: see WarmKeep.drop().

        usable=True goes through WarmArm.rearm, NOT set_keep_warm: the reset
        ran outside the lock long enough for the menu to be unchecked, and
        re-arming anyway leaves the mic open behind an unchecked item (#72).
        """
        dead = self._warm_arm.take()
        self._session.warm.drop(dead)
        if usable:
            self._warm_arm.rearm(self._make_warm_capture)

    def _announce_capture(self, my_gen: int) -> None:
        """Fire on_capture_started at most once per start_recording() cycle
        (issue #43); the generation/recording gate is the staleness defence.
        """
        announce_capture(
            self._lock,
            lambda: self._capture_announced,
            self._stale(my_gen),
            lambda: setattr(self, "_capture_announced", True),
            self.on_capture_started,
        )

    def set_keep_warm(self, enabled: bool) -> None:
        """Keep the input stream open between recordings (issue #66) -- the
        MENU's path, and the only caller that gets to state the request.

        Arming is serialised in WarmArm (issue #72 adversarial
        finding 1): #72 gave it worker- and press-side caller threads, and a
        lock-free check-then-act could start two captures and orphan
        the loser outside WarmKeep, where terminate_all() drives
        PortAudio out from under it.
        """
        self._warm_arm.set_enabled(
            self._make_warm_capture,
            lambda rate: setattr(self, "_capture_rate", rate),
            enabled,
        )

    def _make_warm_capture(self) -> WarmCapture:
        """The capture both arming paths build (menu and re-arm)."""
        return WarmCapture(self._open_warm_stream, self.chunk_size)

    def _open_warm_stream(self) -> tuple[_Stream, int]:
        """One open for WarmCapture's thread. Never overlaps a per-press
        worker's native open on the shared PyAudio (issue #16)."""
        return open_warm_capture_stream(self, pyaudio)

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
        refresh_recorder_session(self, previous)
        # Keep-warm does not re-arm itself after a stuck-read drop (issue #72).
        self._session.warm.recover(self._rearm_warm)
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
        if not warm_press_gate(
            self._lock,
            self._session,
            lambda: (self.recording, self._warm, self.recording_thread),
            find_default_input_device,
            self._rearm_warm,
        ):
            with self._lock:  # every last_error write holds _lock (issue #55)
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
                # The session records the worker BEFORE it runs (issue #72
                # adversarial review, finding 1): from here on the registry
                # can answer "may a worker be inside PortAudio?" for itself,
                # so a later press cannot terminate the engine underneath this
                # one after it is displaced from recording_thread by its own
                # successor.
                self._session.note_worker(thread)

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
        cleanup_recorder_session(self)

    def __del__(self):
        self.cleanup()
