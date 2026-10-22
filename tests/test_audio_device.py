"""Unit tests for kuiskaus.audio_device (issue #72).

No hardware and no CoreAudio framework: the autouse ``coreaudio_stub``
fixture owns the bridge, so a developer's real default input device can
never leak into these assertions.
"""

import threading
from unittest.mock import MagicMock

from kuiskaus.audio_device import PyAudioRegistry
from kuiskaus.audio_retry import check_superseded, registry_for
from tests.support_audio_device import _find_device, _registry
from tests.support_audio_recorder import _live_default_input


def test_open_survives_a_hostile_bridge(coreaudio_stub):
    """__init__'s session open binds the instance even when every CoreAudio
    call raises: no baseline, no crash (the retry loop re-binds later)."""
    coreaudio_stub.AudioObjectGetPropertyData.side_effect = OSError("storm")
    registry, pa, _ = _registry()

    assert registry.open(_find_device) is True
    assert registry.pyaudio is pa
    assert registry.device_id is None


# ---------------------------------------------------------------------------
# The registry: every instance is reachable by terminate_all()
# ---------------------------------------------------------------------------


def test_terminate_all_terminates_every_instance_and_clears():
    registry, first, module = _registry()
    second = MagicMock(name="PyAudioInstance-2")
    registry.register(first)
    registry.register(second)
    registry.register(second)  # idempotent -- no double-terminate

    registry.terminate_all()

    first.terminate.assert_called_once()
    second.terminate.assert_called_once()
    registry.terminate_all()  # nothing left to terminate
    second.terminate.assert_called_once()
    assert module.PyAudio.call_count == 0  # terminate_all never constructs


def test_terminate_all_survives_a_failing_instance():
    """One broken terminate() must not strand the instances after it --
    a single live instance keeps PortAudio's init count above zero."""
    registry, bad, _ = _registry()
    good = MagicMock(name="PyAudioInstance-good")
    bad.terminate.side_effect = OSError("PortAudio still busy")
    registry.register(bad)
    registry.register(good)

    registry.terminate_all()

    good.terminate.assert_called_once()


def test_terminate_unregisters_so_the_instance_is_not_terminated_twice():
    registry, pa, _ = _registry()
    registry.register(pa)
    registry.terminate(pa)
    pa.terminate.assert_called_once()
    registry.terminate_all()
    pa.terminate.assert_called_once()


def test_register_stamps_the_owning_registry():
    """The worker's reset is handed an instance, not the session, so the
    instance must be able to find its registry (issue #72)."""
    registry, pa, _ = _registry()
    registry.register(pa)
    assert registry_for(pa) is registry
    assert registry_for(MagicMock(name="unowned")) is None
    assert registry_for(None) is None


def test_open_binds_the_instance_and_the_live_device(coreaudio_stub):
    _live_default_input(coreaudio_stub, 55)
    registry, pa, module = _registry()

    assert registry.open(_find_device) is True
    assert registry.pyaudio is pa
    assert registry.device_id == 55
    assert registry.input_device_index == 3
    assert module.PyAudio.call_count == 1
    assert registry_for(pa) is registry


def test_open_failure_leaves_session_unbound():
    module = MagicMock(name="pyaudio-module")
    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))
    registry = PyAudioRegistry(module)

    assert registry.open(_find_device) is False
    assert registry.pyaudio is None
    assert registry.input_device_index is None
    assert registry.device_id is None


# ---------------------------------------------------------------------------
# The reset
# ---------------------------------------------------------------------------


def test_reset_is_skipped_without_live_evidence(coreaudio_stub):
    """The no-reset fast path: a transient CoreAudio failure must not cost
    a session rebuild on every press."""
    registry, pa, module = _registry()
    registry.open(_find_device)

    result = registry.refresh(_find_device)

    assert result.busy is False
    assert result.pyaudio is pa
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # only the original open()


def test_moved_default_rebuilds_exactly_one_instance(coreaudio_stub):
    registry, stale, module = _registry()
    registry.open(_find_device)  # baseline: whatever the stub said then
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]  # the reset's one construction
    _live_default_input(coreaudio_stub, 77)

    result = registry.refresh(_find_device)

    stale.terminate.assert_called_once()
    assert result.pyaudio is fresh
    assert result.input_device_index == 3
    assert registry.device_id == 77
    # The whole point of issue #72: one rebuild, not one per attempt.
    assert module.PyAudio.call_count == 2


def test_reset_resolves_the_device_index_on_the_fresh_instance(coreaudio_stub):
    """Issue #72 AC: index resolved on the fresh instance ONLY. The
    pre-#72efresh resolved against the stale cached instance (whose
    PortAudio snapshot was the bug) -- so the resolver must be handed the
    newly constructed PyAudio, never the displaced one."""
    registry, stale, module = _registry()
    registry.open(_find_device)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]
    _live_default_input(coreaudio_stub, 77)
    seen: list[object] = []

    def find(pa):
        seen.append(pa)
        return 3

    result = registry.refresh(find)

    assert result.busy is False
    # First call binds the ORIGINAL open; the reset's resolution is the
    # last one and must name the fresh instance, not the stale one.
    assert seen[-1] is fresh
    assert stale not in seen[1:]
    assert result.input_device_index == 3


def test_reset_when_engine_never_comes_back_leaves_session_unbound(coreaudio_stub):
    registry, _, module = _registry()
    registry.open(_find_device)
    module.PyAudio = MagicMock(side_effect=OSError("Pa_Initialize failed"))
    _live_default_input(coreaudio_stub, 77)

    result = registry.refresh(_find_device)

    assert result.pyaudio is None
    assert result.input_device_index is None
    assert result.busy is False
    assert registry.device_id is None


def test_warm_stream_that_will_not_stop_fails_the_press(coreaudio_stub):
    """A warm capture that would not stop means PortAudio is still up, so
    the session is left whole rather than half-reset -- the next press
    asks again instead of streaming into a stale device."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    warm = MagicMock(name="warm")
    warm.stop.return_value = False

    result = registry.refresh(_find_device, warm)

    assert result.busy is True
    assert result.pyaudio is pa
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1


def test_warm_capture_is_stopped_then_restarted(coreaudio_stub):
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio.side_effect = [fresh]
    warm = MagicMock(name="warm")
    warm.stop.return_value = True
    restart = MagicMock(name="restart-warm")

    result = registry.refresh(_find_device, warm, restart)

    warm.stop.assert_called_once()  # BEFORE terminating: closing the last
    # stream is what releases PortAudio's per-stream init reference
    pa.terminate.assert_called_once()
    restart.assert_called_once()
    assert result.busy is False
    assert result.pyaudio is not pa


def test_reset_reports_busy_rather_than_terminating_under_a_live_worker(
    coreaudio_stub,
):
    """terminate_all() under a thread that may be inside pa.open() or
    stream.read() is the state cleanup() refuses to touch (issue #37
    task-c), and the pre-#72 poll never reached it because the reset never
    ran. Busy, and the session stays whole and dirty (issue #72)."""
    registry, pa, module = _registry()
    registry.open(_find_device)
    _live_default_input(coreaudio_stub, 77)
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    try:
        result = registry.refresh(_find_device, None, None, lambda: (worker,))
    finally:
        release.set()
        worker.join(timeout=5.0)

    assert result.busy is True
    assert result.pyaudio is pa  # nothing torn down
    pa.terminate.assert_not_called()
    assert module.PyAudio.call_count == 1  # ... and nothing rebuilt


# ---------------------------------------------------------------------------
# Adoption and the supersede check
# ---------------------------------------------------------------------------


def _adopt(registry, new_pa, stream, superseded, stream_open, capture_rate=16000):
    published = {}
    ok = registry.adopt(
        threading.Lock(),
        lambda: superseded[0],
        lambda rate: published.__setitem__("rate", rate),
        lambda: stream_open[0],
        new_pa,
        stream,
        capture_rate,
        2,
        0.0,
        4,
    )
    return ok, published


def test_superseded_adopt_closes_the_abandoned_stream_outside_the_lock():
    """closing a stream is a native call and this
    lock is the one stop_recording() needs on the hotkey release. The close is
    deferred past the release (the #60 discipline the rest of the file follows
    for terminate()); running it inside would let a wedged Pa_CloseStream hold
    the release hostage."""
    registry, live, _ = _registry()
    registry.open(_find_device)
    lock = threading.Lock()
    free_at_close: list[bool] = []

    stream = MagicMock(name="stream")

    def close() -> None:
        acquired = lock.acquire(blocking=False)
        free_at_close.append(acquired)
        if acquired:
            lock.release()

    stream.close.side_effect = close
    ok = registry.adopt(
        lock,
        lambda: True,  # superseded: the stream is abandoned, so it gets closed
        lambda _rate: None,
        lambda: False,
        live,
        stream,
        16000,
        2,
        0.0,
        4,
    )

    assert ok is False
    assert stream.close.call_count == 1  # it did run...
    assert free_at_close == [True]  # ... with the lock released


def test_adopt_publishes_instance_and_rate():
    registry, live, _ = _registry()
    registry.open(_find_device)
    stream = MagicMock(name="stream")

    ok, published = _adopt(registry, live, stream, [False], [False])

    assert ok is True
    assert registry.pyaudio is live
    assert published["rate"] == 16000
    # The retry reused the live instance, so nothing may be terminated --
    # the stream is open on it (issue #72 invariant).
    live.terminate.assert_not_called()
    stream.close.assert_not_called()


def test_adopt_disposes_the_instance_a_reset_displaced():
    registry, stale, module = _registry()
    registry.open(_find_device)
    fresh = MagicMock(name="PyAudioInstance-fresh")
    module.PyAudio = MagicMock(return_value=fresh)
    registry.register(fresh)

    ok, _ = _adopt(registry, fresh, MagicMock(name="stream"), [False], [False])

    assert ok is True
    stale.terminate.assert_called_once()


def test_superseded_adopt_closes_stream_and_spares_the_live_session():
    registry, live, _ = _registry()
    registry.open(_find_device)
    stream = MagicMock(name="stream")

    ok, _ = _adopt(registry, live, stream, [True], [False])

    assert ok is False
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    live.terminate.assert_not_called()  # a later press still needs it


def test_superseded_adopt_disposes_an_untracked_fresh_instance():
    registry, _, _ = _registry()
    registry.open(_find_device)
    orphan = MagicMock(name="PyAudioInstance-orphan")

    ok, _ = _adopt(registry, orphan, MagicMock(name="stream"), [True], [False])

    assert ok is False
    orphan.terminate.assert_called_once()


def test_superseded_adopt_disposes_the_registered_instance_the_worker_built():
    """The flow that actually happens (issue #72): attempt_open_once
    registers the instance it constructs, so an abandoned instance is IN the
    registry while the session stays unbound -- registry_for(None) is None
    for the next press and refresh() can never reach it. Left alone it
    holds PortAudio's init count above zero for the life of the process:
    the frozen device list this module exists to clear."""
    registry, worker_pa, _ = _registry()
    registry.register(worker_pa)  # what attempt_open_once did
    assert registry.pyaudio is None  # the session was never bound

    ok, _ = _adopt(registry, worker_pa, MagicMock(name="stream"), [True], [False])

    assert ok is False
    worker_pa.terminate.assert_called_once()
    registry.terminate_all()  # forgotten: cleanup must not terminate twice
    worker_pa.terminate.assert_called_once()


def test_superseded_adopt_keeps_an_orphan_it_could_not_terminate():
    """Same rule as terminate_all(), applied to the orphan adopt() abandons:
    an instance whose terminate() raised is still live and still holds
    PortAudio's init count, so it stays on the list for the next attempt
    instead of becoming unreachable (issue #72)."""
    registry, worker_pa, _ = _registry()
    registry.register(worker_pa)
    worker_pa.terminate.side_effect = OSError("Pa_Terminate failed")

    ok, _ = _adopt(registry, worker_pa, MagicMock(name="stream"), [True], [False])

    assert ok is False
    worker_pa.terminate.assert_called_once()
    registry.terminate_all()  # still reachable: the quit path retries it
    assert worker_pa.terminate.call_count == 2


def test_check_superseded_reports_and_logs(capsys):
    assert check_superseded(threading.Lock(), lambda: True, 2, 4, 0.0) is True
    assert check_superseded(threading.Lock(), lambda: False, 2, 4, 0.0) is False
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[audio.retry]")
    ]
    assert len(lines) == 1
    assert "action=abort" in lines[0]
