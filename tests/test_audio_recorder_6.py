"""test_audio_recorder tests, part 6 of 6 (issue #70)."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.support_audio_recorder import (
    _drain_worker_queue,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
    _warm_recorder,
)


# ---------------------------------------------------------------------------
# Resampling to 16 kHz (issue #55)
# ---------------------------------------------------------------------------
def test_resample_helper_48000_to_16000(audio_recorder_module):
    """Issue #55: resample_to_target converts a 48 kHz mono float32 array
    to a 16 kHz one with length int(N * 16000 / 48000), normalized values
    preserved in range."""
    from kuiskaus.audio_resample import resample_to_target

    n = 48000  # 1 second at 48 kHz
    audio = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    result = resample_to_target(audio, 48000)
    assert len(result) == int(n * 16000 / 48000)  # 16000
    assert result.dtype == np.float32
    assert result.min() >= -1.0 and result.max() <= 1.0


def test_resample_helper_24000_to_16000(audio_recorder_module):
    """Issue #55: 24 kHz -> 16 kHz (AirPods Pro) resamples correctly."""
    from kuiskaus.audio_resample import resample_to_target

    n = 24000  # 1 second at 24 kHz
    audio = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    result = resample_to_target(audio, 24000)
    assert len(result) == int(n * 16000 / 24000)  # 16000


def test_resample_helper_16000_is_noop_bit_identical(audio_recorder_module):
    """Issue #55: 16000 Hz input must be returned unchanged (bit-identical,
    no interpolation error introduced)."""
    from kuiskaus.audio_resample import resample_to_target

    n = 16000
    audio = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    result = resample_to_target(audio, 16000)
    assert len(result) == n
    assert np.array_equal(result, audio)  # bit-identical


def test_resample_helper_none_rate_is_noop(audio_recorder_module):
    """Issue #55: a None capture rate (never recorded) is a no-op."""
    from kuiskaus.audio_resample import resample_to_target

    audio = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    result = resample_to_target(audio, None)
    assert len(result) == 100
    assert np.array_equal(result, audio)


def test_resample_helper_non_positive_rate_raises(audio_recorder_module):
    """Issue #55: non-positive capture rates raise ValueError (0 would
    divide by zero, negatives would corrupt the output silently)."""
    from kuiskaus.audio_resample import resample_to_target

    audio = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    with pytest.raises(ValueError, match="capture_rate must be positive"):
        resample_to_target(audio, 0)
    with pytest.raises(ValueError, match="capture_rate must be positive"):
        resample_to_target(audio, -1)


def test_resample_helper_44100_non_trivial_ratio(audio_recorder_module):
    """Issue #55: 44100 -> 16000 is a non-trivial ratio; the length must
    follow int(N * 16000 / 44100)."""
    from kuiskaus.audio_resample import resample_to_target

    n = 44100  # 1 second at 44.1 kHz
    audio = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    result = resample_to_target(audio, 44100)
    assert len(result) == int(n * 16000 / 44100)  # 16000


def test_stop_recording_resamples_48000(audio_recorder_module):
    """Issue #55: a 48 kHz capture must return 16 kHz-equivalent length
    from stop_recording() -- the downstream transcriber contract."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 48000.0)
    release_event = threading.Event()
    # 1024 frames at 48 kHz = 48000 Hz worth of data in one chunk.
    frame = (np.zeros(1024, dtype=np.int16)).tobytes()
    frame_yielded = threading.Event()

    def fake_read(*_args, **_kwargs):
        if not frame_yielded.is_set():
            frame_yielded.set()
            return frame
        release_event.wait(timeout=10.0)
        return frame

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert _wait_until(lambda: not recorder.audio_queue.empty(), timeout=5.0)

    drained = _drain_worker_queue(recorder)
    release_event.set()

    # The drained bytes are one int16 frame at 48 kHz; resample to 16 kHz.
    arr = np.frombuffer(drained, dtype=np.int16).astype(np.float32) / 32768.0
    from kuiskaus.audio_resample import resample_to_target

    result = resample_to_target(arr, recorder.capture_rate)
    # 1024 samples at 48 kHz -> int(1024 * 16000 / 48000) = 341 samples.
    assert len(result) == int(1024 * 16000 / 48000)
    assert result.dtype == np.float32


def test_stop_recording_resamples_24000(audio_recorder_module):
    """Issue #55: a 24 kHz capture (AirPods Pro) must return 16 kHz-
    equivalent length from stop_recording()."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 24000.0)
    release_event = threading.Event()
    frame = (np.zeros(1024, dtype=np.int16)).tobytes()
    frame_yielded = threading.Event()

    def fake_read(*_args, **_kwargs):
        if not frame_yielded.is_set():
            frame_yielded.set()
            return frame
        release_event.wait(timeout=10.0)
        return frame

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert _wait_until(lambda: not recorder.audio_queue.empty(), timeout=5.0)

    drained = _drain_worker_queue(recorder)
    release_event.set()

    # The drained bytes are one int16 frame at 24 kHz; resample to 16 kHz.
    arr = np.frombuffer(drained, dtype=np.int16).astype(np.float32) / 32768.0
    from kuiskaus.audio_resample import resample_to_target

    result = resample_to_target(arr, recorder.capture_rate)
    # 1024 samples at 24 kHz -> int(1024 * 16000 / 24000) = 682 samples.
    assert len(result) == int(1024 * 16000 / 24000)
    assert result.dtype == np.float32


def test_stop_recording_16000_noop_bit_identical(audio_recorder_module):
    """Issue #55: a 16 kHz capture must return the same sample count and
    bit-identical values (resample skipped entirely, no interpolation)."""
    module = audio_recorder_module
    pa1 = _make_pyaudio_instance(0, 16000.0)
    release_event = threading.Event()
    # A non-trivial pattern so bit-identity is meaningful.
    pattern = np.linspace(-1.0, 1.0, 1024, dtype=np.int16)
    frame = pattern.tobytes()
    frame_yielded = threading.Event()

    def fake_read(*_args, **_kwargs):
        if not frame_yielded.is_set():
            frame_yielded.set()
            return frame
        release_event.wait(timeout=10.0)
        return frame

    stream = MagicMock(name="stream")
    stream.read.side_effect = fake_read
    pa1.open.return_value = stream

    recorder = _make_recorder(module, init_probe=pa1)
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.pyaudio is pa1, timeout=5.0)
    assert _wait_until(lambda: not recorder.audio_queue.empty(), timeout=5.0)

    drained = _drain_worker_queue(recorder)
    release_event.set()

    # The drained bytes are one int16 frame at 16 kHz; resample is a no-op
    # (rate == 16000), so the output must be bit-identical to the
    # int16->float32 normalize.
    arr = np.frombuffer(drained, dtype=np.int16).astype(np.float32) / 32768.0
    from kuiskaus.audio_resample import resample_to_target

    result = resample_to_target(arr, recorder.capture_rate)
    # 1024 samples at 16 kHz -> 1024 samples, unchanged.
    assert len(result) == 1024
    assert result.dtype == np.float32
    expected = pattern.astype(np.float32) / 32768.0
    assert np.array_equal(result, expected)  # bit-identical


def test_warm_press_spawns_no_worker_and_returns_preroll_audio(audio_recorder_module):
    module = audio_recorder_module
    release = threading.Event()
    recorder, pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)
    assert _wait_until(lambda: len(recorder._warm._ring) > 0)
    opens_before = pa.open.call_count

    announced = threading.Event()
    recorder.on_capture_started = announced.set
    assert recorder.start_recording() is True
    assert recorder.recording_thread is None  # no per-press worker
    assert announced.wait(timeout=2.0)

    audio = recorder.stop_recording()
    assert pa.open.call_count == opens_before  # nothing opened on the press path
    assert audio.size > 0
    assert audio.dtype == np.float32
    assert recorder.recording is False
    release.set()


def test_warm_press_during_live_recording_is_refused(audio_recorder_module):
    module = audio_recorder_module
    release = threading.Event()
    recorder, _pa = _warm_recorder(module, _streaming_stream(release))
    assert _wait_until(lambda: recorder._warm.ready)
    assert recorder.start_recording() is True
    assert recorder.start_recording() is False
    recorder.stop_recording()
    release.set()


def test_warm_press_before_ready_reports_error_and_next_press_works(
    audio_recorder_module,
):
    module = audio_recorder_module
    pa = _make_pyaudio_instance(0, rate=16000.0)
    open_gate = threading.Event()
    release = threading.Event()

    def slow_open(**_kwargs):
        open_gate.wait(timeout=10.0)
        return _streaming_stream(release)

    pa.open.side_effect = slow_open
    recorder = _make_recorder(module, init_probe=pa)
    recorder.set_keep_warm(True)

    assert recorder.start_recording() is True  # admitted, never refused
    audio = recorder.stop_recording()
    assert audio.size == 0
    assert recorder.last_error is not None
    assert "busy" in recorder.last_error.lower()

    open_gate.set()
    assert _wait_until(lambda: recorder._warm.ready)
    assert recorder.start_recording() is True
    assert _wait_until(lambda: recorder.audio_queue.qsize() > 0)
    assert recorder.stop_recording().size > 0
    release.set()


def test_disabling_keep_warm_stops_the_background_stream(audio_recorder_module):
    module = audio_recorder_module
    release = threading.Event()
    stream = _streaming_stream(release)
    recorder, _pa = _warm_recorder(module, stream)
    assert _wait_until(lambda: recorder._warm.ready)
    warm = recorder._warm

    recorder.set_keep_warm(False)
    assert recorder._warm is None
    assert warm.ready is False
    stream.close.assert_called()
    release.set()


def test_cleanup_stops_warm_stream(audio_recorder_module):
    module = audio_recorder_module
    release = threading.Event()
    stream = _streaming_stream(release)
    recorder, _pa = _warm_recorder(module, stream)
    assert _wait_until(lambda: recorder._warm.ready)
    recorder.cleanup()
    stream.close.assert_called()
    release.set()
