"""Retry policy (attempt budget, backoff schedule, errno classification),
structured per-attempt logging, one-shot open attempt, and a best-effort
PyAudio terminate helper (issue #37).

Extracted from audio_recorder.py. The recorder's stream-lifecycle state
management and the call sites that invoke terminate_quietly remain in
audio_recorder.py.
"""

import time
from collections.abc import Callable
from types import ModuleType
from typing import Literal

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
) -> tuple["pyaudio.PyAudio | None", "pyaudio.Stream | None", "Exception | None"]:
    """One attempt: construct a fresh PyAudio(), re-resolve the
    default input device against it, and open the stream.

    Returns (pa, stream, None) on success. Returns (None, None,
    error) on any failure, with the failed PyAudio already
    terminated internally and NOT returned in the tuple -- a failed
    attempt's fresh instance owns no stream and was never adopted
    into self.pyaudio, so direct termination is always safe (issue
    #37 task-c) and pa is only ever returned alongside its stream,
    which makes the "don't use a failed attempt's pa" contract
    structural rather than documented.

    Transient failures (PyAudio() construction failure, OSError from
    device enumeration or open()) are returned as the error so the
    loop continues; a RuntimeError from device lookup (no input
    device found at all -- persistent state) is distinguished by
    type at the call site, which aborts the loop.

    The contract for the call site is: ``stream is not None``
    (success) implies ``pa is not None``; ``stream is None``
    implies the attempt failed and ``error`` is set.
    """
    pa: pyaudio.PyAudio
    try:
        pa = pa_module.PyAudio()
    except Exception as construct_error:  # noqa: BLE001 - PyAudio()
        # construction wraps PortAudio's Pa_Initialize(), whose
        # failure modes aren't documented as a narrow exception
        # set. Transient, like an open() OSError: wrap in OSError
        # (errno None) so the loop's RuntimeError check -- reserved
        # for persistent device-lookup failures -- stays unambiguous.
        log_retry_attempt(attempt, max_attempts, attempt_start, None, "open")
        return None, None, OSError(construct_error)

    try:
        device_index = find_default_input_device(pa)
    except OSError as device_error:
        # A coreaudiod storm can make device enumeration itself
        # raise OSError -9986 (issue #37 lens review HIGH #3):
        # transient, same treatment as an open() OSError.
        log_retry_attempt(
            attempt, max_attempts, attempt_start, device_error.errno, "open"
        )
        terminate_quietly(pa)
        return None, None, device_error
    except RuntimeError as device_error:
        # find_default_input_device's own documented failure (no
        # input device found at all) -- persistent state.
        log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
        terminate_quietly(pa)
        return None, None, device_error

    try:
        stream = pa.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )
    except OSError as open_error:
        log_retry_attempt(
            attempt, max_attempts, attempt_start, open_error.errno, "open"
        )
        terminate_quietly(pa)
        return None, None, open_error

    return pa, stream, None


def terminate_quietly(pa: "pyaudio.PyAudio") -> None:
    """Best-effort PyAudio() teardown; a terminate() failure must not
    propagate."""
    try:
        pa.terminate()
    except OSError as e:
        print(f"Error terminating PyAudio instance: {e}")


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
