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

# Attempt 1 runs immediately; attempts 2..MAX_ATTEMPTS are each preceded by a
# sleep drawn from RETRY_BACKOFF_SECONDS.
MAX_ATTEMPTS = 4

# Three sleeps between four attempts, <=1.05s total sleep budget. Spans the
# 300-1500ms coreaudiod-resettle window reported for macOS 26 Tahoe
# stale-object storms. Not yet validated against a real storm -- tunable via
# AudioRecorder(retry_backoff_seconds=...) so operational retuning from real
# per-attempt log data doesn't require a code change.
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.15, 0.30, 0.60)

# pyaudio.paInternalError (-9986). Tahoe coreaudiod stale-object storms
# surface as this generic code, so it is the one errno whose message adds the
# killall-coreaudiod hint -- a heuristic, since -9986 is not exclusively the
# storm signature, but the most actionable safe advice available without
# parsing PortAudio's stderr (not reachable through pyaudio's exception
# surface). Other OSError errnos keep the generic message.
PA_INTERNAL_ERROR_ERRNO = -9986


def format_microphone_error(error: BaseException) -> str:
    """The last_error text for a failed microphone open.

    errno == PA_INTERNAL_ERROR_ERRNO gets the killall-coreaudiod hint;
    everything else -- RuntimeError from device lookup, any other errno --
    keeps the generic message. The foreign text is bounded to 120 chars:
    every branch here reaches the menubar status item (through
    ``microphone_unavailable``), and a native exception can carry an
    arbitrarily long PortAudio/CoreAudio description.
    """
    text = str(error)[:120]
    if isinstance(error, OSError) and error.errno == PA_INTERNAL_ERROR_ERRNO:
        return (
            f"Microphone unavailable ({text}). CoreAudio may be in a bad "
            "state; try 'sudo killall coreaudiod' in Terminal."
        )
    return f"Microphone unavailable: {text}"


def microphone_unavailable(last_error: BaseException | None) -> str:
    """What an exhausted retry loop leaves in ``recorder.last_error``.

    The None case is the common one on a device PortAudio still lists but
    cannot open: every attempt returned no stream without raising, so there
    is no error to name.
    """
    if last_error is None:
        return "Microphone unavailable: unknown error"
    return format_microphone_error(last_error)


# The press that could not start, and the word for why. It lives beside the
# other microphone text the user reads in the menubar's status line (the two
# above, and the restart/not-settling verdicts in audio_device_reset) so the
# three cannot drift into different sentences for one symptom -- the warm
# path's refusals do not run through press_refusal_reason(), so a shared
# constant is the only thing keeping them in step.
MICROPHONE_BUSY = "microphone busy — recording did not start"


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
    """One attempt: reuse ``existing_pa`` when given (the #72 loop always
    gives it, so no attempt ever leaves a second live instance behind), else
    construct one, and open the stream at the device's native rate (issue
    #55), falling back to ``sample_rate`` when the query fails or reports
    something unusable.

    A constructed instance is registered immediately -- PortAudio's init count
    is reference-counted, so an instance nobody tracks is an instance
    terminate_all() can never reach.

    Returns (pa, stream, None, capture_rate) on success, capture_rate being
    the rate actually passed to ``pa.open()``; (None, None, error, None) on
    failure. An instance the attempt itself constructed is terminated
    internally on failure and not returned; ``existing_pa`` is never
    terminated -- the call site owns it.

    Transient failures (construction, enumeration, open) come back as the
    error so the loop continues; a RuntimeError from device lookup means there
    is no input device at all -- persistent -- and the call site aborts on
    that type. ``stream is not None`` implies ``pa`` and ``capture_rate`` are
    both set.
    """
    owns_pa = existing_pa is None
    pa: pyaudio.PyAudio | None
    if owns_pa:
        try:
            pa = pa_module.PyAudio()
        except Exception as construct_error:  # noqa: BLE001 - PyAudio()
            # construction wraps PortAudio's Pa_Initialize(), whose failure
            # modes aren't documented as a narrow exception set. Transient,
            # like an open() OSError: wrapped in OSError (errno None) so the
            # loop's RuntimeError check -- reserved for persistent
            # device-lookup failures -- stays unambiguous.
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "open")
            return None, None, OSError(construct_error), None
        if registry is not None:
            registry.register(pa)
    else:
        pa = existing_pa
    assert pa is not None

    if existing_device_index is not None:
        # The live instance's device was already resolved (at construction or
        # by an earlier attempt); skip the re-resolution.
        device_index = existing_device_index
    else:
        try:
            device_index = find_default_input_device(pa)
        except OSError as device_error:
            # A coreaudiod storm can make enumeration itself raise -9986:
            # transient, same treatment as an open() OSError.
            log_retry_attempt(
                attempt, max_attempts, attempt_start, device_error.errno, "open"
            )
            if owns_pa:
                _dispose_attempt_instance(pa, registry)
            return None, None, device_error, None
        except RuntimeError as device_error:
            # No input device found at all -- persistent state.
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            if owns_pa:
                _dispose_attempt_instance(pa, registry)
            return None, None, device_error, None
        if registry is not None:
            # Hand the resolved index back -- paired with the instance it was
            # resolved against -- or it dies with this attempt and every later
            # press pays a fresh enumeration.
            registry.bind_index(device_index, pa)

    # Open at the device's native rate: opening at a rate CoreAudio does not
    # use triggers the renegotiation behind Tahoe's -9986 storm.
    effective_rate = sample_rate
    try:
        # The device actually being opened, not the default: the default can
        # move between the index resolution and this query (AirPods
        # disconnecting mid-open), which would otherwise record a rate for a
        # different device than the one opened.
        device_info = pa.get_device_info_by_index(device_index)
        default_sample_rate = device_info["defaultSampleRate"]
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as query_error:
        # Bounded message: a pathological CoreAudio property value must not
        # dump an unbounded (possibly binary) repr into stderr.
        print(
            f"Native rate query failed ({str(query_error)[:120]!r}); "
            f"falling back to {sample_rate}"
        )
    else:
        if default_sample_rate is None or default_sample_rate <= 0:
            # Log the corrupted-but-present report so an operator debugging a
            # persistent -9986 storm can tell "opened at the 16000 fallback
            # because the device reported an invalid rate" from "opened at
            # native".
            print(
                f"Native rate {default_sample_rate!r} invalid (<=0); "
                f"falling back to {sample_rate}"
            )
        elif not (4000 <= default_sample_rate <= 192000):
            # A wildly out-of-range value is a corrupted device report, not a
            # real rate -- fall back and say so.
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

    # The rate rides the return so the caller adopts the exact value passed to
    # pa.open() under _lock with the stream, keeping open-and-record rate
    # atomic without shared mutable state.
    return pa, stream, None, effective_rate


def terminate_quietly(pa: "pyaudio.PyAudio") -> bool:
    """Best-effort PyAudio() teardown; a terminate() failure must not
    propagate.

    The verdict is the point: an instance whose ``terminate()`` raised is
    still live and still holds PortAudio's init count, so the caller must keep
    it reachable rather than treat it as gone.
    """
    try:
        pa.terminate()
    except Exception as e:  # noqa: BLE001 - PortAudio teardown boundary, logged
        # Not just OSError: Pa_Terminate runs its own cleanup and PyAudio
        # raises whatever its wrapper produces, and an escaping exception here
        # would abort terminate_all() before it reached the remaining
        # instances -- a live instance nobody can reach any more. Bounded
        # like the rest of the foreign text: a native description can be
        # arbitrarily long and this prints on cleanup paths.
        print(f"Error terminating PyAudio instance: {str(e)[:120]}")
        return False
    return True


def _dispose_attempt_instance(
    pa: "pyaudio.PyAudio", registry: "PyAudioRegistry | None"
) -> None:
    """Terminate an instance the attempt itself constructed, and untrack it: a
    terminated instance left in the registry would be terminated twice, and an
    *untracked* live one is what keeps PortAudio's init count above zero."""
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
    errno={e|"-"} action={sleep|open|adopt|abort}``. Lets a bug reporter's
    real-world coreaudiod storm timing be read back from application logs
    post-ship -- host-repro at rest could not reproduce the storm, so this is
    the validation channel for the retry budget's design envelope.
    """
    elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
    errno_field = errno if errno is not None else "-"
    print(
        f"[audio.retry] attempt={attempt}/{max_attempts} "
        f"elapsed_ms={elapsed_ms} errno={errno_field} action={action}"
    )


def check_superseded(
    lock: threading.Lock,
    is_superseded: Callable[[], bool],
    attempt: int,
    max_attempts: int,
    attempt_start: float,
) -> bool:
    """True (and logged) when this retry sequence lost its race: a newer
    generation took over, or recording was released.

    Shared by the pre-sleep and post-sleep abort sites so the two cannot drift
    apart. The predicate is evaluated under the caller's lock, which is the
    single arbiter of that state.
    """
    with lock:
        if is_superseded():
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            return True
        return False


def close_stream_quietly(stream: "pyaudio.Stream") -> None:
    """Best-effort stream teardown; close failures must not propagate.

    The same net as ``terminate_quietly``: stop_stream()/close() are native
    PortAudio calls behind PyAudio's wrapper, which raises whatever it
    produces -- an OSError-only net let a wrapper RuntimeError escape into
    the caller's cleanup path (a dropped or abandoned stream), where it
    aborts the rest of that path over a stream nobody will use again.
    """
    try:
        stream.stop_stream()
        stream.close()
    except Exception as e:  # noqa: BLE001 - native teardown boundary, logged
        print(f"Error closing stream: {str(e)[:120]}")


# Upper bound on how long a new worker waits for an orphaned predecessor stuck
# in a native open() (issue #60). CoreAudio opens have been observed blocking
# ~24 s; past this the new worker proceeds rather than hang forever.
PREVIOUS_WORKER_WAIT_SECONDS = 30.0


def worker_alive(thread: "threading.Thread | None") -> bool:
    """Lock-free worker-thread liveness predicate (issue #16). The thread
    attribute is assigned once per generation (not mutated) and is_alive() on
    a dead thread is idempotent, so this is safe to call outside
    recorder._lock."""
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

    Returns the PyAudio instance used (a fresh one if ``pa`` was None) so the
    caller can adopt it, the stream, and the rate it was opened at. The
    instance is tracked in ``registry`` either way, so a warm session is never
    the one live instance terminate_all() cannot see.
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
