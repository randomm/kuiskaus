"""The recording worker's session lifecycle (issue #72).

Three procedures, all of them the worker's own view of the device session:
reset it at the head of a press, name the worker the gate must ask about, and
fail the press when the session could not be prepared.

Split from ``kuiskaus.audio_warm_arm`` -- which is the warm-capture arming and
the press gate, decided under the recorder lock -- because these run OUTSIDE
that lock (issue #60: they reach CoreAudio and PortAudio, which can block for
seconds while ``stop_recording()`` needs the lock), and because the hazard
they carry is different: what they write back is the session pair, and a pair
written while another thread is rebuilding it names an instance PortAudio has
already torn down.

``RecorderState`` stays where it is (the protocol describes the recorder as
the warm/press component sees it) and is imported for annotations only, so
this module never imports ``audio_warm_arm`` at runtime -- the one direction
that does is ``audio_warm_arm`` calling ``refresh_recorder_session`` from the
warm capture's open.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from kuiskaus.audio_retry import find_default_input_device

if TYPE_CHECKING:
    from kuiskaus.audio_warm_arm import RecorderState


def current_worker(recorder: RecorderState) -> threading.Thread | None:
    """The worker named at CALL time, read under _lock: a name captured
    before a long join is stale by gate time -- a finished W1 must not mask a
    live W2 (issue #72).
    """
    with recorder._lock:
        return recorder.recording_thread


def refresh_recorder_session(
    recorder: RecorderState, blocked: threading.Thread | None = None
) -> None:
    """Live-default device reset (issue #72) at the top of the worker.

    Native calls run OUTSIDE _lock (issue #60): they can block and
    stop_recording() needs the lock. One worker runs this at a time;
    ``blocked`` is the worker just joined, and the gate also consults the
    recorder's CURRENT worker at gate time -- which is why the
    name travels as a callable rather than as the thread.

    The session is asked for its own reset, not looked up through the instance
    it happens to be holding: a half-finished teardown unbinds ``pyaudio``
    (terminate_all() terminated the live instance and a second one refused)
    while that instance stays registered, and that is exactly the state where
    the reset is the only thing that can finish the job. Resolving the session
    from a None instance answers "no session here" and leaves PortAudio
    initialised on a frozen device list -- the #72 symptom, back inside the
    fix, on the default (non-keep-warm) path.

    A ``locked`` deferral writes nothing back (see below): the session belongs
    to whoever holds ``_reset_lock``, and the pair this result snapshots may
    predate that thread's teardown.
    """
    result = recorder._session.refresh(
        find_default_input_device,
        blocked_workers=lambda: (blocked, current_worker(recorder)),
    )
    if result.locked:
        # Nothing is written back. Another thread holds ``_reset_lock`` and is
        # mutating ``pyaudio`` / ``_indices`` right now, so the pair this
        # result carries was sampled mid-teardown: ``_dispose`` terminates the
        # instance BEFORE ``forget``/``unbind`` run, and a ``snapshot()`` taken
        # in that window names an instance ``Pa_Terminate()`` has already been
        # called on while its index still resolves -- the (corpse, live index)
        # pairing ``DeviceIndexBinding`` exists to make unrepresentable.
        # Writing it back clobbers the rebuild the holder is about to publish,
        # and because the holder's ``open()`` left ``last_reset`` agreeing with
        # the live device, every later press then takes the dedupe arm of
        # ``_settled_or_refused`` and re-publishes the corpse: the mic dead
        # until restart, through the path this commit added.
        print(
            "[audio.retry] device reset deferred: another thread is resetting "
            "the session; its write-back stands, not this snapshot."
        )
        return
    if result.busy:
        # The press proceeds against the stale session (the only safe move
        # while a thread may be inside a native call), but the deferral has to
        # be visible in the same log stream as the open attempts.
        print(
            "[audio.retry] device reset deferred: a worker may still hold "
            "PortAudio; session stays dirty for the next press."
        )
    recorder.publish_session(result.pyaudio, result.input_device_index)


def fail_worker_session(
    recorder: RecorderState, my_gen: int, error: BaseException | None
) -> None:
    """Generation-gated teardown for a worker that never got a stream.

    Shared by the exhausted retry loop and both of the worker's guards: either
    way the press is over, and "recording" must not outlive the thread that
    was recording it. The message goes through the registry's wrapper so a
    device the session already knows is gone reads differently from a busy
    microphone.
    """
    with recorder._lock:
        if recorder._generation != my_gen:
            return
        recorder.last_error = recorder._session.unavailable_reason(error)
        recorder.recording = False
        recorder.stream = None
        recorder.recording_thread = None
