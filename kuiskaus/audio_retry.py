"""Retry policy (attempt budget, backoff schedule, errno classification),
structured per-attempt logging, and a best-effort PyAudio terminate
helper (issue #37).

Extracted from audio_recorder.py. The recorder's stream-lifecycle state
management and the call sites that invoke terminate_quietly remain in
audio_recorder.py.
"""

import time
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
