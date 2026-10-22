"""Retry policy (attempt budget, backoff schedule, errno classification),
structured per-attempt logging, one-shot open attempt, and a best-effort
PyAudio terminate helper (issue #37).

Extracted from audio_recorder.py. The recorder's stream-lifecycle state
management and the call sites that invoke terminate_quietly remain in
audio_recorder.py.
"""

import math
import threading
import time
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import pyaudio

if TYPE_CHECKING:
    from kuiskaus.audio_device import PyAudioRegistry

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
# killall-coreaudiod hint in format_microphone_error() is a heuristic --
# -9986 is not exclusively the Tahoe storm signature, but it is the most
# actionable generally-safe advice available without parsing PortAudio's
# stderr warnings, which are not accessible through pyaudio's exception
# surface. Other OSError errnos (e.g. -9985 paDeviceUnavailable, -9997
# paInvalidDevice) deliberately keep the generic message.
PA_INTERNAL_ERROR_ERRNO = -9986


def format_microphone_error(error: BaseException) -> str:
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


def microphone_unavailable(last_error: BaseException | None) -> str:
    """What an exhausted retry loop leaves in ``recorder.last_error``.

    The None case is the common one on a device PortAudio still lists but
    cannot open: every attempt returned no stream without raising, so there
    is no error to name. It lives with the retry policy that produced
    ``last_error`` rather than in the recorder, which has no part in
    deciding what the failure is called.
    """
    if last_error is None:
        return "Microphone unavailable: unknown error"
    return format_microphone_error(last_error)


def attempt_open_once(
    pa_module: ModuleType,
    format: int,
    channels: int,
    sample_rate: int,
    chunk_size: int,
    find_default_input_device: Callable[["pyaudio.PyAudio"], int],
    max_attempts: int,
    attempt: int,
    attempt_start: float,
    existing_pa: "pyaudio.PyAudio | None" = None,
    existing_device_index: int | None = None,
    registry: "PyAudioRegistry | None" = None,
) -> tuple[
    "pyaudio.PyAudio | None", "pyaudio.Stream | None", "Exception | None", int | None
]:
    """One attempt: construct a fresh PyAudio() (or reuse ``existing_pa``
    when provided -- issue #72's retry loop always provides it, so the
    loop never leaves a second live instance behind) and open the stream
    at the device's native sample rate (issue #55).

    A constructed instance is registered with ``registry`` immediately --
    PortAudio's init count is reference-counted, so an instance nobody
    tracks is an instance terminate_all() can never reach.

    The native rate is queried from ``pa.get_device_info_by_index(
    device_index)["defaultSampleRate"]``. If the query fails or the
    value is missing/invalid, the provided ``sample_rate`` (16000)
    is used as the fallback.

    Returns (pa, stream, None, capture_rate) on success where
    ``capture_rate`` is the int rate actually passed to ``pa.open()``.
    Returns (None, None, error, None) on any failure. With a constructed
    (fresh) instance, the failed PyAudio is terminated internally and
    NOT returned in the tuple; with ``existing_pa`` provided, the
    provided instance is NEVER terminated on failure: it is the
    recorder's cached self.pyaudio, which the call site owns.

    Transient failures (PyAudio() construction failure, OSError from
    device enumeration or open()) are returned as the error so the
    loop continues; a RuntimeError from device lookup (no input
    device found at all -- persistent state) is distinguished by
    type at the call site, which aborts the loop.

    The contract for the call site is: ``stream is not None``
    (success) implies ``pa is not None`` and ``capture_rate is not
    None``; ``stream is None`` implies the attempt failed and
    ``error`` is set.
    """
    owns_pa = existing_pa is None
    pa: pyaudio.PyAudio | None
    if owns_pa:
        try:
            pa = pa_module.PyAudio()
        except Exception as construct_error:  # noqa: BLE001 - PyAudio()
            # construction wraps PortAudio's Pa_Initialize(), whose
            # failure modes aren't documented as a narrow exception
            # set. Transient, like an open() OSError: wrap in OSError
            # (errno None) so the loop's RuntimeError check -- reserved
            # for persistent device-lookup failures -- stays unambiguous.
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "open")
            return None, None, OSError(construct_error), None
        if registry is not None:
            registry.register(pa)
    else:
        # Issue #42 attempt 1: reuse the recorder's cached instance; the
        # call site owns it and keeps it on failure.
        pa = existing_pa
    assert pa is not None

    if existing_device_index is not None:
        # Issue #72: the loop reuses the live instance, whose device was
        # already resolved (at construction or by an earlier attempt);
        # skip the re-resolution.
        device_index = existing_device_index
    else:
        try:
            device_index = find_default_input_device(pa)
        except OSError as device_error:
            # A coreaudiod storm can make device enumeration itself
            # raise OSError -9986 (issue #37 lens review HIGH #3):
            # transient, same treatment as an open() OSError.
            log_retry_attempt(
                attempt, max_attempts, attempt_start, device_error.errno, "open"
            )
            if owns_pa:
                _dispose_attempt_instance(pa, registry)
            return None, None, device_error, None
        except RuntimeError as device_error:
            # find_default_input_device's own documented failure (no
            # input device found at all) -- persistent state.
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            if owns_pa:
                _dispose_attempt_instance(pa, registry)
            return None, None, device_error, None
        if registry is not None:
            # The session had no usable index: hand the resolved one back,
            # or it dies with this attempt and every later press pays a
            # fresh enumeration (issue #72). It is handed back PAIRED with
            # the instance it was resolved against, so a binding left by a
            # disposed attempt can never be read as if it still named a
            # device (issue #72 review, round 1).
            registry.bind_index(device_index, pa)

    # Issue #55: query the device's native sample rate so the stream
    # is opened at the rate CoreAudio actually uses, avoiding the
    # sample-rate renegotiation that triggers Tahoe's -9986 storm.
    # Falls back to the provided ``sample_rate`` (16000) when the
    # key is missing, zero, or the call raises (coreaudiod storm).
    effective_rate = sample_rate
    try:
        # Query the device actually being opened (the resolved
        # device_index), not the default input device: the default can
        # move between the index resolution and this query (e.g. AirPods
        # disconnect mid-open), which would otherwise record a rate for
        # a different device than the one opened (TOCTOU, issue #55
        # lens review MEDIUM, security).
        device_info = pa.get_device_info_by_index(device_index)
        default_sample_rate = device_info["defaultSampleRate"]
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as query_error:
        # Missing key, or the query itself raised (coreaudiod storm), or
        # the value is non-numeric. Fall back to the provided
        # ``sample_rate`` (16000); the print is the observability for
        # why a device opened at the fallback. The message is bounded
        # -- a pathological CoreAudio property value must not dump an
        # unbounded (possibly binary) repr into stderr.
        print(
            f"Native rate query failed ({str(query_error)[:120]!r}); "
            f"falling back to {sample_rate}"
        )
    else:
        if default_sample_rate is None or default_sample_rate <= 0:
            # A device that reports a present-but-corrupted (zero/negative)
            # native rate is precisely the failure mode this ticket
            # addresses; log it so an operator debugging a persistent
            # -9986 storm can tell "opened at 16000 fallback because the
            # device reported an invalid rate" from "opened at native".
            print(
                f"Native rate {default_sample_rate!r} invalid (<=0); "
                f"falling back to {sample_rate}"
            )
        elif not (4000 <= default_sample_rate <= 192000):
            # Sanity bound (generous superset of any real input device's
            # native rate): a wildly out-of-range value is a corrupted
            # device report, not a real rate -- fall back and say so.
            print(
                f"Native rate {default_sample_rate!r} out of bounds "
                "(4000..192000); falling back to "
                f"{sample_rate}"
            )
        else:
            effective_rate = int(default_sample_rate)

    try:
        stream = pa.open(
            format=format,
            channels=channels,
            rate=effective_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )
    except OSError as open_error:
        log_retry_attempt(
            attempt, max_attempts, attempt_start, open_error.errno, "open"
        )
        if owns_pa:
            _dispose_attempt_instance(pa, registry)
        return None, None, open_error, None

    # 4-tuple (widened from 3-tuple in issue #55): the successful rate
    # rides the return so the single caller adopts the exact value
    # passed to pa.open() under _lock for stop_recording() resampling,
    # keeping open-and-record rate atomic without shared mutable state.
    return pa, stream, None, effective_rate


def terminate_quietly(pa: "pyaudio.PyAudio") -> bool:
    """Best-effort PyAudio() teardown; a terminate() failure must not
    propagate.

    The verdict is the point (issue #72): an instance whose ``terminate()``
    raised is still live and still holds PortAudio's init count, so the
    caller must keep it reachable rather than treat it as gone.
    """
    try:
        pa.terminate()
    except Exception as e:  # noqa: BLE001 - PortAudio teardown boundary, logged
        # Not just OSError: Pa_Terminate runs its own cleanup and PyAudio
        # raises whatever its wrapper produces, and an escaping exception
        # here would abort terminate_all() before it reached the remaining
        # instances -- a live instance nobody can reach any more, which is
        # the exact frozen-snapshot state this class exists to clear.
        print(f"Error terminating PyAudio instance: {e}")
        return False
    return True


# PortAudio's init count is reference-counted (issue #72): an instance is
# only reachable by terminate_all() through the registry that built it, so
# every registered instance carries a back-reference to its owning
# session. Stamped by PyAudioRegistry.register(), read here so the worker's
# device reset can find the session from the instance it was handed.
_OWNER_ATTRIBUTE = "_kuiskaus_registry"

# Imported lazily inside registry_for(): audio_device imports this module at
# module scope, so a top-level import here would be a cycle.
_owner_stamp_logged = False


def stamp_owner(pa: "pyaudio.PyAudio", registry: "PyAudioRegistry") -> None:
    """Record which session registry built ``pa`` (see _OWNER_ATTRIBUTE).

    An instance that cannot carry the stamp is an instance no worker-side
    reset can ever find its session through, so the failure is said once --
    silently skipping it would leave the frozen-device-list symptom with no
    trace of why the reset never runs.
    """
    global _owner_stamp_logged
    try:
        setattr(pa, _OWNER_ATTRIBUTE, registry)
    except (AttributeError, TypeError) as e:
        if not _owner_stamp_logged:
            _owner_stamp_logged = True
            print(
                f"Cannot stamp the owning session on a PyAudio instance ({e}); "
                "device resets will not find it."
            )


def registry_for(pa: "pyaudio.PyAudio | None") -> "PyAudioRegistry | None":
    """The session registry that built ``pa`` (issue #72), or None.

    None means no registry tracked this instance -- there is no tracked
    set to drive to zero, so the caller leaves the session alone instead
    of half-resetting it. The check is an isinstance against the real
    registry: duck-typing on a private attribute name accepted anything
    that happened to expose a list called ``_instances``, which is a
    promise about a field name rather than about the object.
    """
    if pa is None:
        return None
    owner = getattr(pa, _OWNER_ATTRIBUTE, None)
    if isinstance(owner, _registry_class()):
        return owner
    return None


def _registry_class() -> type:
    """``PyAudioRegistry``, imported on first use (see _OWNER_ATTRIBUTE)."""
    from kuiskaus.audio_device import PyAudioRegistry

    return PyAudioRegistry


def _dispose_attempt_instance(
    pa: "pyaudio.PyAudio", registry: "PyAudioRegistry | None"
) -> None:
    """Terminate an instance the attempt itself constructed, and untrack
    it (issue #72): a terminated instance left in the registry would be
    terminated twice, and an *untracked* live one is what keeps
    PortAudio's init count above zero."""  # fmt: skip
    if registry is None:
        terminate_quietly(pa)
    else:
        registry.terminate(pa)


def log_retry_attempt(
    attempt: int,
    max_attempts: int,
    attempt_start: float,
    errno: int | None,
    action: Literal["sleep", "open", "adopt", "abort"],
) -> None:
    """Emit one structured per-attempt retry log line to stdout.

    Format: ``[audio.retry] attempt={n}/{max_attempts} elapsed_ms={m}
    errno={e|"-"} action={sleep|open|adopt|abort}``. Lets a bug
    reporter's real-world coreaudiod storm timing be read back from
    application logs post-ship (issue #37) -- host-repro at rest could
    not reproduce the storm, so this is the validation channel for the
    retry budget's design envelope.
    """
    elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
    errno_field = errno if errno is not None else "-"
    print(
        f"[audio.retry] attempt={attempt}/{max_attempts} "
        f"elapsed_ms={elapsed_ms} errno={errno_field} action={action}"
    )


def refresh_pyaudio_session(
    pyaudio_instance: "pyaudio.PyAudio | None",
    input_device_index: int | None,
    find_default_input_device: Callable[["pyaudio.PyAudio"], int],
    blocked_workers: "Callable[[], Sequence[threading.Thread | None]] | None" = None,
) -> tuple["pyaudio.PyAudio | None", int | None]:
    """Device-change reset (issue #72), called at the top of the recording
    worker before the retry loop.

    Replaces the issue #42 poll. That poll asked the CACHED instance what
    its default input device was -- but PortAudio snapshots the hardware
    once at Pa_Initialize and never refreshes it, so the comparison could
    not observe a switch and the stale device survived every press. The
    trigger is now a live CoreAudio read (kuiskaus.audio_device), and a
    reset drives PortAudio's init count to zero through the session's
    registry before rebuilding: a single untracked live instance is enough
    to make the next construction replay the frozen device list.

    An instance with no owning registry is left untouched -- either the
    session was never bound (the retry loop builds and registers it) or the
    instance came from outside this session, and in neither case is there a
    tracked set to drive to zero.

    Returns (pyaudio_instance, input_device_index) -- possibly both
    replaced. The caller is responsible for any lock scoping.

    ``blocked_workers`` names the threads whose native PortAudio call this
    reset must not run under (issue #72 review): the workers the caller
    joined and may still have alive. The names resolve through the
    callable at GATE time -- a thread named before a long join may have
    finished while a newer worker became current (finding 2). Busy is
    reported by leaving the session exactly as it is, so the next press
    reads the live device again rather than PortAudio going away under a
    thread inside pa.open().
    """
    registry = registry_for(pyaudio_instance)
    if registry is None:
        return pyaudio_instance, input_device_index
    result = registry.refresh(
        find_default_input_device, blocked_workers=blocked_workers
    )
    if result.busy:
        # The press proceeds against the stale session (the only safe
        # move while a thread may be inside a native call), but the
        # deferral must be observable in the same log stream as the
        # open attempts themselves (#72 review).
        print(
            "[audio.retry] device reset deferred: a worker may still hold "
            "PortAudio; session stays dirty for the next press."
        )
    return result.pyaudio, result.input_device_index


def check_superseded(
    lock: threading.Lock,
    is_superseded: Callable[[], bool],
    attempt: int,
    max_attempts: int,
    attempt_start: float,
) -> bool:
    """True (and logged) when this retry sequence lost its race.

    A newer generation took over, or recording was released. Shared by the
    pre-sleep and post-sleep abort sites so the two cannot drift apart. The
    predicate is evaluated under the caller's lock, which is the single
    arbiter of that state.
    """
    with lock:
        if is_superseded():
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            return True
        return False


def close_stream_quietly(stream: "pyaudio.Stream") -> None:
    """Best-effort stream teardown; close failures must not propagate."""
    try:
        stream.stop_stream()
        stream.close()
    except OSError as e:
        print(f"Error closing stream: {e}")


# Upper bound on how long a new worker waits for an orphaned predecessor
# stuck in a native open() (issue #60). CoreAudio opens have been observed
# blocking ~24 s; past this the new worker proceeds rather than hang forever.
PREVIOUS_WORKER_WAIT_SECONDS = 30.0


def worker_alive(thread: "threading.Thread | None") -> bool:
    """Lock-free worker-thread liveness predicate (issue #16). The
    thread attribute is assigned once per generation (not mutated)
    and is_alive() on a dead thread is idempotent, so this is safe
    to call outside recorder._lock."""
    return thread is not None and thread.is_alive()


def find_default_input_device(pa: "pyaudio.PyAudio") -> int:
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


def open_warm_stream(
    pa_module: ModuleType,
    pa: "pyaudio.PyAudio | None",
    device_index: int | None,
    format: int,
    channels: int,
    sample_rate: int,
    chunk_size: int,
    registry: "PyAudioRegistry | None" = None,
) -> tuple["pyaudio.PyAudio", "pyaudio.Stream", int]:
    """One attempt for the keep-warm stream (issue #66); raises on failure.

    Returns the PyAudio instance used (a fresh one if ``pa`` was None) so
    the caller can adopt it, the stream, and the rate it was opened at.
    The instance is tracked in ``registry`` either way, so a warm session
    is never the one live instance terminate_all() cannot see (issue #72).
    """
    new_pa, stream, error, rate = attempt_open_once(
        pa_module,
        format,
        channels,
        sample_rate,
        chunk_size,
        find_default_input_device,
        1,
        1,
        time.monotonic(),
        existing_pa=pa,
        existing_device_index=device_index,
        registry=registry,
    )
    if stream is None or new_pa is None or rate is None:
        raise error or OSError("warm open failed")
    return new_pa, stream, rate


def validate_retry_config(
    max_attempts: int, retry_backoff_seconds: Sequence[float]
) -> None:
    """Reject retry settings the open loop cannot honour."""
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


def log_stop(chunks: int, start_monotonic: float | None, reason: str) -> None:
    """Emit the structured ``[audio.stop]`` line (issue #40)."""
    duration_ms = (
        int((time.monotonic() - start_monotonic) * 1000)
        if start_monotonic is not None
        else 0
    )
    print(f"[audio.stop] chunks={chunks} duration_ms={duration_ms} reason={reason}")
