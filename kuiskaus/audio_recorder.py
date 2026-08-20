import queue
import threading

import numpy as np
import pyaudio


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,  # Whisper expects 16kHz
        chunk_size: int = 1024,
        channels: int = 1,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16

        self.pyaudio = pyaudio.PyAudio()
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

        # Find the default input device
        self.input_device_index = self._find_default_input_device(self.pyaudio)

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

    def _open_stream_with_retry(self, my_gen: int) -> "pyaudio.Stream | None":
        """Open the input stream, retrying once with a fresh PyAudio instance.

        Happy path: opens with ``self.pyaudio`` / ``self.input_device_index``
        exactly as before, no construction or reassignment.

        Retry path (only on a first-attempt OSError): constructs a fresh
        ``pyaudio.PyAudio()``, re-resolves the input device, and attempts
        open() once more. Deliberately no time budget -- pyaudio.open() is
        an uninterruptible blocking C call, so a timeout claim would be
        unenforceable and a thread-plus-join wrapper would leave a
        permanently stuck daemon thread (issue #16).

        Returns the opened stream, or None if both attempts failed
        (``last_error`` is set in that case, gated on ``my_gen`` still
        being current) or the generation was superseded mid-retry.
        """
        try:
            return self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
            )
        except OSError as first_error:
            print(f"Microphone open failed ({first_error}); retrying once")

        retry_pyaudio = pyaudio.PyAudio()
        try:
            retry_device_index = self._find_default_input_device(retry_pyaudio)
            stream = retry_pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=retry_device_index,
                frames_per_buffer=self.chunk_size,
            )
        except (OSError, RuntimeError) as retry_error:
            # RuntimeError is _find_default_input_device's own documented
            # failure (no input device found at all) -- without catching
            # it here too, it would propagate out of this method, out of
            # _recording_worker (no surrounding try/except there), and
            # kill the daemon thread silently: last_error would stay None
            # and self.recording would stay True, reproducing the original
            # bug for one press (self-healing on the next, via stale-state
            # recovery in start_recording()).
            # The fresh instance owns no stream: safe to terminate directly.
            retry_pyaudio.terminate()
            with self._lock:
                if self._generation == my_gen:
                    self.last_error = f"Microphone unavailable: {retry_error}"
                    self.recording = False
                    self.stream = None
                    self.recording_thread = None
            return None

        with self._lock:
            if self._generation != my_gen:
                # Superseded while retrying: this stream belongs to no one.
                self._close_stream_quietly(stream)
                retry_pyaudio.terminate()
                return None
            # Adopt only after the retried open() succeeded. The superseded
            # old instance is deliberately NOT terminated here -- it may
            # still hold a stream, and terminate() with a live stream is
            # unsafe. Accepted trade-off: one leaked PortAudio instance per
            # successful retry.
            print("Adopting fresh PyAudio instance after microphone retry succeeded")
            self.pyaudio = retry_pyaudio
            self.input_device_index = retry_device_index

        return stream

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
            thread.join(timeout=1.0)
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
        is unsafe. Note: after a successful retry adoption, self.pyaudio
        may no longer be the instance this recorder started with.
        """
        if self.recording:
            self.stop_recording()

        with self._lock:
            recording = self.recording
            thread = self.recording_thread

        if recording or (thread is not None and thread.is_alive()):
            print(
                "Recording worker still active at cleanup; skipping PyAudio.terminate()"
            )
            return

        self.pyaudio.terminate()

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()
