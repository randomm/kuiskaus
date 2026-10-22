"""The unnamed orphan worker (issue #72 adversarial review, CRITICAL).

Split into its own module (test_audio_recorder_7.py is at the file cap).
This drives the exact four-press sequence the review reproduced through the
public API only, on the real ``AudioRecorder`` and the real
``PyAudioRegistry``:

  1. Press A starts W1, which wedges inside ``pa.open()`` -- issue #60's
     orphan. ``stop_recording()`` reports stuck-open; ``recording_thread``
     still names W1.
  2. Press B starts W2 with ``previous=W1``. It joins W1 for
     ``PREVIOUS_WORKER_WAIT_SECONDS`` and then PROCEEDS BY DESIGN -- a wedge
     must not wedge later presses.
  3. W2 completes and clears ``recording_thread``. W1 is still alive, and
     now nobody can name it.
  4. Press C asks the gate what may be inside PortAudio. A gate that knows
     only the caller's two names is told "nothing", and ``Pa_Terminate``
     frees the state W1 is still entering.

The registry's worker ledger is what closes step 4, so the assertion is the
one the review's probe produced: no terminate while a worker it started is
still inside ``pa.open()``.
"""

import threading

from tests.support_audio_recorder import (
    _cleanup_recorder,
    _live_default_input,
    _make_pyaudio_instance,
    _make_recorder,
    _streaming_stream,
    _wait_until,
)


def test_four_presses_cannot_terminate_under_a_worker_that_lost_its_name(
    audio_recorder_module, coreaudio_stub, monkeypatch
):
    """The CRITICAL: the reset gate named the two threads its caller passed,
    and step 3 takes the wedged worker out of that inventory. Busy, not
    terminated -- and the press still runs."""
    module = audio_recorder_module
    # The join budget is what makes step 2 slow, not what makes it wrong:
    # shortening it only reaches the same state sooner.
    monkeypatch.setattr(module, "PREVIOUS_WORKER_WAIT_SECONDS", 0.05)

    wedged = threading.Event()
    release = threading.Event()
    live = {"in_open": 0, "calls": 0}

    def _open(*_args, **_kwargs):
        live["calls"] += 1
        if live["calls"] == 1:
            # W1 is deterministically parked inside the native open for as
            # long as the test needs it there -- the only state in which a
            # terminate is undefined behaviour rather than merely rude.
            live["in_open"] += 1
            wedged.wait(timeout=10.0)
            live["in_open"] -= 1
            raise OSError("wedged open released by test")
        return _streaming_stream(release)

    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.side_effect = _open
    recorder = _make_recorder(
        module, init_probe=pa, max_attempts=2, retry_backoff_seconds=(0.01,)
    )

    try:
        # --- press A: W1 wedges in pa.open() -----------------------------
        assert recorder.start_recording() is True
        worker_1 = recorder.recording_thread
        assert worker_1 is not None
        assert _wait_until(lambda: live["in_open"] == 1, timeout=5.0)
        assert recorder.stop_recording().size == 0  # stuck-open
        assert recorder.recording_thread is worker_1

        # --- press B: W2 joins W1 and proceeds anyway (issue #60) --------
        assert recorder.start_recording() is True
        worker_2 = recorder.recording_thread
        assert worker_2 is not None and worker_2 is not worker_1
        release.set()  # let W2's read end so it can complete
        assert _wait_until(lambda: not worker_2.is_alive(), timeout=8.0)

        # --- step 3: the recorder has lost W1's name, not W1 --------------
        assert recorder.recording_thread is None
        assert worker_1.is_alive()
        assert live["in_open"] == 1

        # --- press C: the gate is told nothing is alive -------------------
        # The default moved, so this press really wants the reset.
        _live_default_input(coreaudio_stub, 77)
        assert recorder.start_recording() is True
        worker_3 = recorder.recording_thread
        assert worker_3 is not None
        # Either the reset was refused (this worker opens on the live
        # instance, so pa.open runs again) or it terminated underneath W1.
        # Waiting on the disjunction keeps the assertion below the one that
        # decides the test.
        assert _wait_until(
            lambda: pa.terminate.called or live["calls"] >= 3, timeout=8.0
        )

        # The violation the review measured: pa.terminate() firing while
        # another thread sits inside pa.open() on that same instance.
        pa.terminate.assert_not_called()
        assert live["in_open"] == 1  # W1 never left the native call
        # Busy, not half-reset: the press proceeds on the session it has.
        assert recorder.pyaudio is pa
        assert _wait_until(lambda: not worker_3.is_alive(), timeout=8.0)
        # ... and the press did not pay for it with a second live instance.
        assert module.pyaudio.PyAudio.call_count == 1
    finally:
        wedged.set()
        release.set()
        _cleanup_recorder(recorder)


def test_wedged_worker_stops_blocking_once_its_open_returns(
    audio_recorder_module, coreaudio_stub, monkeypatch
):
    """The ledger is a gate, not a verdict (same rule ``WarmKeep`` follows):
    once the wedged open returns and the worker exits, the deferred reset
    runs and the stale session is actually replaced."""
    module = audio_recorder_module
    monkeypatch.setattr(module, "PREVIOUS_WORKER_WAIT_SECONDS", 0.05)

    wedged = threading.Event()
    release = threading.Event()
    live = {"in_open": 0, "calls": 0}

    def _open(*_args, **_kwargs):
        live["calls"] += 1
        if live["calls"] == 1:
            live["in_open"] += 1
            wedged.wait(timeout=10.0)
            live["in_open"] -= 1
            raise OSError("wedged open released by test")
        return _streaming_stream(release)

    pa = _make_pyaudio_instance(0, rate=16000.0)
    pa.open.side_effect = _open
    rebuild = _make_pyaudio_instance(5, rate=16000.0)
    rebuild.open.return_value = _streaming_stream(release)
    # Baseline 77 at construction, so press A needs no reset and gets
    # straight to the native open; the move to 88 is what makes the later
    # presses want one.
    _live_default_input(coreaudio_stub, 77)
    recorder = _make_recorder(
        module,
        rebuild,
        init_probe=pa,
        extra_pa=rebuild,
        max_attempts=2,
        retry_backoff_seconds=(0.01,),
    )

    try:
        assert recorder.start_recording() is True
        worker = recorder.recording_thread
        assert worker is not None
        assert _wait_until(lambda: live["in_open"] == 1, timeout=5.0)
        assert recorder.stop_recording().size == 0

        # Deferred: the live worker owns the session, nothing is torn down.
        _live_default_input(coreaudio_stub, 88)
        assert recorder.start_recording() is True
        later = recorder.recording_thread
        assert later is not None
        release.set()
        assert _wait_until(lambda: not later.is_alive(), timeout=8.0)
        pa.terminate.assert_not_called()

        # The wedge clears, so the next press gets the reset it was refused.
        wedged.set()
        assert _wait_until(lambda: not worker.is_alive(), timeout=8.0)
        assert recorder.start_recording() is True
        final = recorder.recording_thread
        assert final is not None
        assert _wait_until(lambda: recorder.pyaudio is rebuild, timeout=8.0)
        pa.terminate.assert_called_once()

        release.set()
        assert _wait_until(lambda: not final.is_alive(), timeout=8.0)
    finally:
        wedged.set()
        release.set()
        _cleanup_recorder(recorder)
