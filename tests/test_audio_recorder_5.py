"""test_audio_recorder tests, part 5 of 6 (issue #70)."""

import threading
from unittest.mock import MagicMock

import pytest

from tests.support_audio_recorder import (
    _blocking_read_stream,
    _blocking_read_then_block,
    _blocking_stream,
    _capture_callback_recorder,
    _make_pyaudio_instance,
    _make_recorder,
    _wait_until,
)


def test_on_capture_started_flag_resets_per_start_recording_cycle(
    audio_recorder_module,
):
    """The once-flag is per start_recording() cycle: a second cycle on the
    same recorder fires the callback again."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)
    release1 = threading.Event()
    stream1 = MagicMock(name="stream-1")
    stream1.read.side_effect = _blocking_read_then_block(b"\x00" * 2048, release1)
    pa1.open.return_value = stream1
    # Cycle 2 reuses the cache (device unchanged); extra_pa=pa2 is the
    # factory's fallback instance (its device-info mock is stable, so no
    # poll reconstruction is triggered).
    pa2 = _make_pyaudio_instance(1)
    recorder = _make_recorder(
        module, init_probe=pa1, extra_pa=pa2, on_capture_started=callback
    )
    assert recorder.start_recording() is True
    assert _wait_until(lambda: callback.call_count == 1, timeout=5.0)
    release1.set()
    thread1 = recorder.recording_thread
    assert thread1 is not None
    thread1.join(timeout=10.0)
    assert not thread1.is_alive()
    assert callback.call_count == 1

    # Second cycle: attempt 1 reuses the cache (pa1); repoint its open()
    # at stream2. The flag was reset by start_recording, so the callback
    # fires again.
    release2 = threading.Event()
    stream2 = MagicMock(name="stream-2")
    stream2.read.side_effect = _blocking_read_then_block(b"\x03" * 2048, release2)
    pa1.open.return_value = stream2
    assert recorder.start_recording() is True
    # Wait for the callback before capturing (post-adoption idiom -- the
    # worker is then blocked on release2 inside read()).
    assert _wait_until(lambda: callback.call_count == 2, timeout=5.0)
    thread2 = recorder.recording_thread
    assert thread2 is not None

    release2.set()
    thread2.join(timeout=10.0)
    assert not thread2.is_alive()
    assert callback.call_count == 2


def test_on_capture_started_not_called_on_empty_capture(audio_recorder_module):
    """Release-before-first-read (the race-lost case, issue #40's
    newly-surfaced outcome): stop_recording() may land before the worker's
    first stream.read() returns (or even before it is called), so the
    queue is empty and the callback must never fire.

    The stream.read mock blocks on a test-owned event (via _blocking_stream)
    rather than returning an unconfigured MagicMock default, so the test is
    deterministic regardless of which side of the race wins: if the worker
    breaks before read() (the intended race-lost path), or if it enters
    read() and blocks there, no MagicMock ever enters audio_queue and
    b"".join() in stop_recording() never sees a non-bytes item."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    pa1.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # stop_recording() clears self.recording under lock and joins the
    # worker; if the worker is still blocked in read() the join times out
    # (stuck-open path), which is fine -- the queue is still empty.
    result = recorder.stop_recording()

    # Unblock any worker that entered read() before the gate check, then
    # let it finish its teardown. If the worker broke out before read()
    # (the intended race-lost case) it has already exited and recording_thread
    # is None; in that case this is a no-op.
    release_event.set()
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=10.0)
        assert not thread.is_alive()

    assert not recorder.recording
    assert result.size == 0
    assert callback.call_count == 0


def test_on_capture_started_skipped_when_generation_superseded(
    audio_recorder_module,
):
    """A newer recording superseding this worker between read() and the
    callback firing (generation bump under lock) must suppress the
    callback -- the generation gate is rechecked after read() returns."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)

    def fake_read(*_args, **_kwargs):
        # Simulate a newer generation taking over while the worker is in
        # flight between read() and the callback. This is the exact race
        # the issue's "generation gate" requirement exists to close.
        with recorder._lock:
            recorder._generation += 1
            recorder.recording = True  # the newer generation is recording
        return b"\x00" * 2048

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # The worker's read loop breaks immediately after the first read
    # (the generation no longer matches my_gen) and its post-loop
    # teardown is generation-gated, so it never clears recording for
    # this generation; the only deterministic observable is the stream
    # closure below. Wait on that instead of the transient thread handle
    # (lens review HIGH #1/#4 read-before-start race).
    assert _wait_until(lambda: stream.close.called, timeout=5.0)

    assert callback.call_count == 0
    # The stale worker must still have closed the stream it owned.
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()


def test_on_capture_started_skipped_when_recording_false(audio_recorder_module):
    """stop_recording() landing between the worker's read() returning and
    the callback firing (recording False under lock) must suppress the
    callback -- the recording-True gate is rechecked after read()."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)

    def fake_read(*_args, **_kwargs):
        # Simulate a release landing between read() and the callback.
        with recorder._lock:
            recorder.recording = False
        return b"\x00" * 2048

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1, on_capture_started=callback)
    assert recorder.start_recording() is True
    # The worker's read loop breaks on the next iteration (recording is
    # False); the callback is suppressed this time and the loop exits.
    # The worker can clear recording_thread before a capture, so assert
    # on observable state instead (lens review HIGH #1/#4 read-before-
    # start race).
    assert _wait_until(lambda: recorder.recording is False, timeout=5.0)
    thread = recorder.recording_thread
    if thread is not None:
        thread.join(timeout=10.0)

    assert callback.call_count == 0
    # The worker still tears down the stream it owned.
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()


def test_on_capture_started_callback_exception_does_not_stop_worker(
    audio_recorder_module, capsys
):
    """A raising callback must not kill the read loop: the broad guard
    (callback boundary, # noqa: BLE001) logs and continues. The next
    read() still succeeds and the callback is NOT retried (the flag was
    set before the raise)."""
    module = audio_recorder_module

    def boom():
        raise RuntimeError("callback failure")

    recorder, thread, release = _capture_callback_recorder(module, boom, b"\x00" * 2048)

    # The flag is set before the callback runs, so the worker continues
    # the loop after the raise without retrying; the second read blocks
    # on release.
    assert _wait_until(lambda: recorder._capture_announced is True, timeout=5.0)

    release.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    captured = capsys.readouterr()
    assert "on_capture_started callback raised" in captured.out


def test_on_capture_started_default_none_is_backwards_compatible(
    audio_recorder_module,
):
    """AudioRecorder() without on_capture_started never calls anything;
    recording behaves exactly as before the callback existed. The
    flag is set (as a bookkeeping no-op) but no callback runs."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    release_event = threading.Event()
    pa1.open.return_value = _blocking_stream(OSError("stop the loop"), release_event)
    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.on_capture_started is None
    assert recorder._capture_announced is False

    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    # The worker is deterministically alive (blocked in read()). Wait for
    # adoption so the post-loop assertion on the flag is race-free.
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert thread.is_alive()

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_on_capture_started_fires_after_retry_adoption(audio_recorder_module):
    """The callback fires on the FIRST successful non-empty read ACROSS
    ALL attempts, i.e. after retry adoption lands and the adopted stream
    delivers its first bytes -- not once per retry attempt. Attempt 1
    never reaches a read() (open() fails), so no callback fires from it.
    A non-empty read on the adopted (attempt-2) stream fires it exactly
    once."""
    module = audio_recorder_module
    callback = MagicMock()
    pa1 = _make_pyaudio_instance(0)
    pa1.open.side_effect = OSError("attempt 1 failed")
    pa_retry = _make_pyaudio_instance(1)
    release_event = threading.Event()
    pa_retry.open.return_value = _blocking_read_stream(b"\x00" * 2048, release_event)

    recorder = _make_recorder(
        module,
        pa_retry,
        init_probe=pa1,
        max_attempts=2,
        on_capture_started=callback,
    )
    assert recorder.start_recording() is True
    # Wait for the retry adoption, then for the callback to fire on the
    # adopted stream's first non-empty read; by then the worker is
    # blocked on release_event inside read(), so the capture is
    # race-free (post-adoption idiom).
    assert _wait_until(lambda: recorder.pyaudio is pa_retry, timeout=5.0)
    assert _wait_until(lambda: callback.call_count == 1, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert callback.call_count == 1  # once total, not once per attempt


# ---------------------------------------------------------------------------
# Native sample rate (issue #55)
# ---------------------------------------------------------------------------
def test_open_stream_uses_native_rate_48000(audio_recorder_module):
    """Issue #55: a device reporting defaultSampleRate=48000.0 must have
    pa.open() called with rate=48000, not 16000."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 48000.0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)

    # The native rate must be used, not the hardcoded 16000.
    assert recorder.capture_rate == 48000
    pa1.open.assert_called_once_with(
        format=pa1.open.call_args.kwargs["format"],
        channels=pa1.open.call_args.kwargs["channels"],
        rate=48000,
        input=True,
        input_device_index=pa1.open.call_args.kwargs["input_device_index"],
        frames_per_buffer=pa1.open.call_args.kwargs["frames_per_buffer"],
    )

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_open_stream_uses_native_rate_24000(audio_recorder_module):
    """Issue #55: AirPods Pro (24000 Hz) must have the stream opened at
    rate=24000, not 16000."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 24000.0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)

    assert recorder.capture_rate == 24000

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_open_stream_falls_back_to_16000_when_no_native_rate(audio_recorder_module):
    """Issue #55: when the device info dict lacks "defaultSampleRate"
    (the existing _make_pyaudio_instance fixture), the stream is opened
    at the fallback rate of 16000 -- no crash, no rate=0."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)  # no defaultSampleRate key
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    # Wait for adoption: the capture_rate is only written when the open
    # succeeds and is adopted, so it is the race-free synchronization.
    assert _wait_until(lambda: recorder.capture_rate == 16000, timeout=5.0)

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_open_stream_falls_back_to_16000_when_rate_query_raises(
    audio_recorder_module, capsys
):
    """Issue #55: when get_default_input_device_info() raises during the
    native-rate query (coreaudiod storm), the stream opens at the fallback
    rate of 16000 and the fallback is printed -- never a silent skip.

    The mock's open-time rate query (get_device_info_by_index -- the
    exact device being opened, issue #55 lens review MEDIUM security)
    raises OSError -- a per-attempt OSError is what the edge-case spec
    describes for a coreaudiod storm mid-recording."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0)
    # The device-index resolution (get_default_input_device_info) still
    # works; only the open-time rate query against the resolved device
    # (get_device_info_by_index) raises OSError -- a coreaudiod storm.
    pa1.get_device_info_by_index.side_effect = OSError("-9986")
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert _wait_until(lambda: recorder.capture_rate == 16000, timeout=5.0)

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    out = capsys.readouterr().out
    assert "falling back to 16000" in out


def test_open_stream_falls_back_to_16000_when_rate_is_nonfinite(audio_recorder_module):
    """Issue #55: a defaultSampleRate of float('inf') (a driver-reported
    sentinel) must be treated as unusable -- the sanity bounds check
    (4000..192000) rejects any non-finite value (float('inf') fails the
    upper bound, float('nan') fails both) -- and fall back to 16000
    without killing the worker."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, float("inf"))
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert _wait_until(lambda: recorder.capture_rate == 16000, timeout=5.0)

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_capture_rate_is_none_before_any_recording(audio_recorder_module):
    """Issue #55: capture_rate is None before any stream has been opened."""
    module = audio_recorder_module
    recorder = _make_recorder(module, _make_pyaudio_instance(0))
    assert recorder.capture_rate is None


def test_capture_rate_accessible_via_property(audio_recorder_module):
    """Issue #55: the capture_rate property returns the int rate set by
    the open path, and is read-only (assignment raises AttributeError)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 44100.0)
    release_event = threading.Event()
    stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    thread = recorder.recording_thread
    assert thread is not None
    assert _wait_until(lambda: recorder.capture_rate == 44100, timeout=5.0)

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()

    with pytest.raises(AttributeError):
        recorder.capture_rate = 99999


def test_native_rate_requeried_on_retry_with_fresh_instance(audio_recorder_module):
    """Issue #55: the native rate is re-queried per attempt against
    whichever pa instance performs the open. A retry on a fresh instance
    with a different native rate uses that instance's rate."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 48000.0)
    pa1.open.side_effect = OSError("attempt 1 failed")
    pa_retry = _make_pyaudio_instance(1, 24000.0)
    release_event = threading.Event()
    retry_stream = _blocking_stream(OSError("stop the loop"), release_event)
    pa_retry.open.return_value = retry_stream

    recorder = _make_recorder(
        module, pa_retry, init_probe=pa1, max_attempts=2, retry_backoff_seconds=(0.01,)
    )
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa_retry, timeout=5.0)
    thread = recorder.recording_thread
    assert thread is not None

    # The retry's fresh instance reported 24000; that's the rate used.
    assert recorder.capture_rate == 24000

    release_event.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
