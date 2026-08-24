import math
import queue
import threading
import time
from collections.abc import Sequence

import numpy as np
import pyaudio

# Bounded attempt count for the microphone-open backoff-and-re-enumerate
# loop (issue #37). Attempt 1 runs immediately; attempts 2..MAX_ATTEMPTS
# are each preceded by a sleep drawn from RETRY_BACKOFF_SECONDS.
MAX_ATTEMPTS = 4

# Three sleeps between four attempts, <=1.05s total sleep budget. Spans
# the 300-1500ms coreaudiod-resettle window reported for macOS 26 Tahoe
# stale-object storms (see issue #37). Not yet validated against a real
# storm -- tunable via AudioRecorder(retry_backoff_seconds=...) so
# operational retuning from real per-attempt log data doesn't require a
# code change.
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.15, 0.30, 0.60)

# pyaudio.paInternalError (-9986): macOS Tahoe coreaudiod stale-object storms
# surface as this generic PortAudio internal-error code (issue #37). The
# killall-coreaudiod hint in _format_microphone_error() is a heuristic --
# -9986 is not exclusively the Tahoe storm signature, but it is the most
# actionable generally-safe advice available without parsing PortAudio's
# stderr warnings, which are not accessible through pyaudio's exception
# surface. Other OSError errnos (e.g. -9985 paDeviceUnavailable, -9997
# paInvalidDevice) deliberately keep the generic message.
PA_INTERNAL_ERROR_ERRNO = -9986


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,  # Whisper expects 16kHz
        chunk_size: int = 1024,
        channels: int = 1,
        max_attempts: int = MAX_ATTEMPTS,
        retry_backoff_seconds: Sequence[float] = RETRY_BACKOFF_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        if max_attempts > 1 and not retry_backoff_seconds:
            raise ValueError(
                "retry_backoff_seconds must be non-empty when max_attempts > 1 "
                "(pass max_attempts=1 for no retries)"
            )
        if any(not math.isfinite(s) or s < 0 for s in retry_backoff_seconds):
            raise ValueError(
                "retry_backoff_seconds must contain only finite, non-negative values"
            )
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        # Covers the full backoff sleep budget plus slack for typical
        # open() latency; a worst-case run of max_attempts slow opens can
        # still exceed this by up to ~0.5s and is surfaced late via the
        # release path -- accepted, see issue #37 Technical Context.
        self._stuck_open_timeout_seconds: float = sum(retry_backoff_seconds) + 0.5

        # No long-lived PyAudio() instance (issue #37 task-c): every
        # recording constructs its own fresh instance in
        # _open_stream_with_retry and adopts it into self.pyaudio only on
        # a successful open(). self.pyaudio is None until the first
        # successful recording.
        self.pyaudio: pyaudio.PyAudio | None = None
        self.stream: pyaudio.Stream | None = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread: threading.Thread | None = None
        self.last_error: str | None = None

        # Concurrency guard (issue #16): protects state sections only, is
        # never held across thread.join()/pyaudio.open(), and is paired
        # with a generation counter so a worker whose recording has been
        # superseded can never clobber a newer generation's state.
        #
        # start_recording()'s admission test gates on thread liveness
        # alone and deliberately omits the `self.recording` conjunct (see
        # start_recording()'s docstring): including it would let a second
        # worker be admitted while a stuck-but-still-alive worker from a
        # prior press was already calling pyaudio.open() on the shared
        # self.pyaudio instance concurrently -- the exact native-level
        # hazard issue #16 exists to close.
        #
        # Consequence: a new generation can only ever be created once no
        # prior worker thread is alive, so the `self._generation == my_gen`
        # checks inside _recording_worker()/_open_stream_with_retry() are
        # unreachable via any real start_recording()/stop_recording()
        # sequence today. They remain correct and unit-tested; keep them
        # as defence-in-depth against a future relaxation of the admission
        # rule above -- do not treat them as load-bearing today, and do
        # not delete them as dead code.
        self._lock = threading.Lock()
        self._generation = 0

        # Validate a default input device exists using a temporary
        # PyAudio() instance, terminated immediately after (issue #37
        # task-c). This is a sanity check only -- the recorder does not
        # depend on this instance surviving; every recording re-resolves
        # the device against its own fresh instance in
        # _open_stream_with_retry. A RuntimeError here (no input device
        # found at all) propagates out of __init__ unchanged.
        probe = pyaudio.PyAudio()
        try:
            self._find_default_input_device(probe)
        finally:
            self._terminate_quietly(probe)

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

    @staticmethod
    def _terminate_quietly(pa: "pyaudio.PyAudio") -> None:
        """Best-effort PyAudio() teardown; a terminate() failure must not
        propagate (mirrors _close_stream_quietly)."""
        try:
            pa.terminate()
        except OSError as e:
            print(f"Error terminating PyAudio instance: {e}")

    @staticmethod
    def _format_microphone_error(error: BaseException) -> str:
        """Build the last_error text for a failed microphone open.

        OSError.errno == PA_INTERNAL_ERROR_ERRNO (paInternalError, -9986)
        gets the killall-coreaudiod hint (issue #37); every other failure
        -- including RuntimeError from device lookup and any other OSError
        errno -- keeps the generic message unchanged.
        """
        if isinstance(error, OSError) and error.errno == PA_INTERNAL_ERROR_ERRNO:
            return (
                f"Microphone unavailable ({error}). CoreAudio may be in a bad "
                "state; try 'sudo killall coreaudiod' in Terminal."
            )
        return f"Microphone unavailable: {error}"

    @staticmethod
    def _log_retry_attempt(
        attempt: int,
        max_attempts: int,
        elapsed_ms: int,
        errno: int | None,
        action: str,
    ) -> None:
        """Emit one structured per-attempt retry log line to stdout.

        Format: ``[audio.retry] attempt={n}/{max_attempts} elapsed_ms={m}
        errno={e|"-"} action={sleep|open|adopt|abort}``. Lets a bug
        reporter's real-world coreaudiod storm timing be read back from
        application logs post-ship (issue #37) -- host-repro at rest could
        not reproduce the storm, so this is the validation channel for the
        retry budget's design envelope.
        """
        errno_field = errno if errno is not None else "-"
        print(
            f"[audio.retry] attempt={attempt}/{max_attempts} "
            f"elapsed_ms={elapsed_ms} errno={errno_field} action={action}"
        )

    def _open_stream_with_retry(self, my_gen: int) -> "pyaudio.Stream | None":
        """Open the input stream, retrying with backoff up to ``self.max_attempts``.

        Every attempt -- including attempt 1 -- constructs a fresh
        ``pyaudio.PyAudio()`` and re-resolves the default input device
        against that fresh instance before calling ``open()`` (issue #37
        task-c). ``self.pyaudio`` and the old ``input_device_index`` cache
        are retired: a device change between recordings (e.g. AirPods
        disconnecting) is picked up on attempt 1 without waiting for a
        first failure, per the issue's Expected Behaviour #3.

        Adoption on success is a lock-scoped local ownership transfer:
        ``old_pyaudio = self.pyaudio; self.pyaudio = pa`` happens under
        ``self._lock``; ``old_pyaudio`` (the previous recording's
        instance, if any) is terminated *outside* the lock, gated on
        ``self.stream is None`` -- the invariant every teardown path in
        this file maintains. A failed attempt's own fresh instance is
        terminated directly at its own failure site (it owns no stream by
        construction of the failure, and was never adopted into
        self.pyaudio), so nothing is left to leak within a single
        recording's own retry sequence; only a genuinely previous
        instance is ever disposed via the local-capture path.

        Deliberately no time budget on open() itself -- pyaudio.open() is
        an uninterruptible blocking C call, so a timeout claim would be
        unenforceable and a thread-plus-join wrapper would leave a
        permanently stuck daemon thread (issue #16).

        Before each attempt after the first, checks ``self._generation ==
        my_gen`` and ``self.recording`` (under ``_lock``); either being
        false aborts the loop early and returns without writing state, so
        a hotkey release mid-backoff does not burn the remaining sleep
        budget.

        A ``RuntimeError`` from device re-resolution (no input device found
        at all) is treated as a persistent failure and aborts the loop
        immediately rather than continuing to burn attempts against it. A
        failure to construct ``pyaudio.PyAudio()`` itself (e.g. a
        ``Pa_Initialize()`` failure during a severe coreaudiod storm) is
        treated as transient, like an ``OSError`` from ``open()``: it costs
        the current attempt and the loop continues.

        Returns the opened stream, or None if all attempts failed
        (``last_error`` is set in that case, gated on ``my_gen`` still
        being current) or the generation/recording state was superseded
        mid-retry.

        Per-attempt structured logging (issue #37): every attempt emits
        exactly one ``_log_retry_attempt`` line. Action ``"sleep"`` marks
        the backoff decision before attempts 2..N; ``"open"`` marks a
        failed ``PyAudio()`` construction or a failed ``open()``;
        ``"adopt"`` marks a successful open; ``"abort"`` marks a
        superseded generation/recording state or a persistent
        RuntimeError from device lookup.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            attempt_start = time.monotonic()

            if attempt > 1:
                with self._lock:
                    if self._generation != my_gen or not self.recording:
                        self._log_retry_attempt(
                            attempt,
                            self.max_attempts,
                            int((time.monotonic() - attempt_start) * 1000),
                            None,
                            "abort",
                        )
                        return None

                self._log_retry_attempt(attempt, self.max_attempts, 0, None, "sleep")
                backoff_index = min(attempt - 2, len(self.retry_backoff_seconds) - 1)
                time.sleep(self.retry_backoff_seconds[backoff_index])

                with self._lock:
                    if self._generation != my_gen or not self.recording:
                        self._log_retry_attempt(
                            attempt,
                            self.max_attempts,
                            int((time.monotonic() - attempt_start) * 1000),
                            None,
                            "abort",
                        )
                        return None

            try:
                pa = pyaudio.PyAudio()
            except Exception as construct_error:  # noqa: BLE001 - PyAudio()
                # construction wraps PortAudio's Pa_Initialize(), whose
                # failure modes aren't documented as a narrow exception
                # set. Treated as transient, like an open() OSError: it
                # costs this attempt and the loop continues, so a
                # construction failure can never propagate out of this
                # method and kill the worker thread uncaught.
                last_error = construct_error
                print(
                    f"PyAudio initialization failed (attempt "
                    f"{attempt}/{self.max_attempts}): {construct_error}"
                )
                self._log_retry_attempt(
                    attempt,
                    self.max_attempts,
                    int((time.monotonic() - attempt_start) * 1000),
                    None,
                    "open",
                )
                continue

            try:
                device_index = self._find_default_input_device(pa)
            except RuntimeError as device_error:
                # _find_default_input_device's own documented failure
                # (no input device found at all) -- persistent state,
                # further attempts won't help. The fresh instance owns
                # no stream: safe to terminate directly.
                self._terminate_quietly(pa)
                last_error = device_error
                self._log_retry_attempt(
                    attempt,
                    self.max_attempts,
                    int((time.monotonic() - attempt_start) * 1000),
                    None,
                    "abort",
                )
                break

            try:
                stream = pa.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                )
            except OSError as open_error:
                last_error = open_error
                # The fresh instance owns no stream: safe to terminate
                # directly -- it was never adopted into self.pyaudio.
                self._terminate_quietly(pa)
                print(
                    f"Microphone open failed (attempt {attempt}/{self.max_attempts}): "
                    f"{open_error}"
                )
                self._log_retry_attempt(
                    attempt,
                    self.max_attempts,
                    int((time.monotonic() - attempt_start) * 1000),
                    open_error.errno,
                    "open",
                )
                continue

            with self._lock:
                if self._generation != my_gen or not self.recording:
                    # Superseded while retrying: this stream belongs to no one.
                    self._close_stream_quietly(stream)
                    self._terminate_quietly(pa)
                    self._log_retry_attempt(
                        attempt,
                        self.max_attempts,
                        int((time.monotonic() - attempt_start) * 1000),
                        None,
                        "abort",
                    )
                    return None
                # Lock-scoped local ownership transfer (issue #37
                # task-c): capture the previous instance locally and
                # reassign under the same lock acquisition that writes
                # self.pyaudio, so a concurrent cleanup() can never
                # observe a torn state.
                old_pyaudio = self.pyaudio
                self.pyaudio = pa

            # Outside the lock: dispose of the previous session's
            # PyAudio() instance, gated on self.stream is None -- the
            # invariant every teardown path in this file maintains.
            # Whoever captures old_pyaudio owns its termination;
            # cleanup() uses the same local-capture pattern, so
            # double-terminate is impossible by construction.
            if old_pyaudio is not None and self.stream is None:
                self._terminate_quietly(old_pyaudio)

            self._log_retry_attempt(
                attempt,
                self.max_attempts,
                int((time.monotonic() - attempt_start) * 1000),
                None,
                "adopt",
            )
            return stream

        with self._lock:
            if self._generation == my_gen:
                self.last_error = (
                    self._format_microphone_error(last_error)
                    if last_error is not None
                    else "Microphone unavailable: unknown error"
                )
                self.recording = False
                self.stream = None
                self.recording_thread = None
        return None

    def _recording_worker(self, my_gen: int) -> None:
        """Worker thread for continuous audio recording.

        Every shared-state write here (failure path, adoption, and
        post-loop teardown) is gated on ``self._generation == my_gen`` so a
        late/stale worker can never clobber a newer recording's state.
        """
        stream = self._open_stream_with_retry(my_gen)
        if stream is None:
            return  # last_error already set (or generation superseded)

        with self._lock:
            if self._generation != my_gen:
                self._close_stream_quietly(stream)
                return
            self.stream = stream
            # Only clear last_error if this generation's session is still
            # active. If self.recording is already False here, the
            # session was ended (e.g. stop_recording()'s stuck-open path
            # already surfaced an error for this generation) before this
            # late open() finally returned; clobbering that error would
            # hide it from a release handler that may have already read
            # it, or race with one about to. The while loop below still
            # breaks immediately since self.recording is False.
            if self.recording:
                self.last_error = None

        while True:
            with self._lock:
                if self._generation != my_gen or not self.recording:
                    break
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except OSError as e:
                # CoreAudio/pyaudio I/O failure: log and stop the recording loop
                print(f"Error recording audio: {e}")
                break

        self._close_stream_quietly(stream)

        with self._lock:
            if self._generation == my_gen:
                self.recording = False
                self.stream = None
                self.recording_thread = None

    def start_recording(self) -> bool:
        """Start recording audio.

        Returns True if a new recording was admitted and a worker spawned,
        False if refused because a worker is already alive and recording.

        Liveness (not ``self.recording``) is the sole refusal signal: after
        stop_recording()'s stuck-open path, self.recording is already False
        while recording_thread may still be physically blocked inside a
        native pyaudio call. Gating refusal on self.recording as well would
        let a second worker call pyaudio.open() on the same shared
        self.pyaudio concurrently with the still-running one -- exactly the
        native-level hazard issue #16 exists to close. Liveness alone is
        race-free: once a thread is observed not-alive it can never become
        alive again, so the immediate reassignment below is always safe.
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
            self.recording = True
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
        """Stop recording and return the audio data as numpy array."""
        with self._lock:
            was_recording = self.recording
            thread = self.recording_thread
            my_gen = self._generation
            if was_recording:
                self.recording = False

        if not was_recording:
            return np.array([], dtype=np.float32)

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

        return np.array([], dtype=np.float32)

    def cleanup(self) -> None:
        """Clean up PyAudio resources.

        terminate() is skipped -- and logged -- if a worker may still be
        alive, since terminate() while another thread holds/uses a stream
        is unsafe. Uses the same lock-scoped local-capture pattern as
        between-recordings adoption in _open_stream_with_retry (issue #37
        task-c): whoever captures self.pyaudio into a local owns its
        termination, so a concurrent adoption and cleanup() can never
        double-terminate the same instance. No separate
        ``_pyaudio_terminated`` flag is needed -- the local capture makes
        double-terminate impossible by construction.
        """
        if not hasattr(self, "recording"):
            # __init__ raised (e.g. invalid retry_backoff_seconds) before
            # state was fully initialized; there is nothing to clean up.
            return

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
            self._terminate_quietly(old_pyaudio)

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()
