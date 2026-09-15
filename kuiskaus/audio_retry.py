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
    existing_pa: "pyaudio.PyAudio | None" = None,
    existing_device_index: int | None = None,
) -> tuple[
    "pyaudio.PyAudio | None", "pyaudio.Stream | None", "Exception | None", int | None
]:
    """One attempt: construct a fresh PyAudio() (or reuse ``existing_pa``
    when provided -- issue #42's attempt 1 reuses the cached instance
    and its cached device index, skipping re-resolution) and open the
    stream at the device's native sample rate (issue #55).

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
    else:
        # Issue #42 attempt 1: reuse the recorder's cached instance; the
        # call site owns it and keeps it on failure.
        pa = existing_pa
    assert pa is not None

    if existing_device_index is not None:
        # Issue #42 attempt 1: the device was already resolved against
        # the cached instance (at construction or by the worker's
        # poll); skip the re-resolution.
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
                terminate_quietly(pa)
            return None, None, device_error, None
        except RuntimeError as device_error:
            # find_default_input_device's own documented failure (no
            # input device found at all) -- persistent state.
            log_retry_attempt(attempt, max_attempts, attempt_start, None, "abort")
            if owns_pa:
                terminate_quietly(pa)
            return None, None, device_error, None

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
            terminate_quietly(pa)
        return None, None, open_error, None

    # 4-tuple (widened from 3-tuple in issue #55): the successful rate
    # rides the return so the single caller adopts the exact value
    # passed to pa.open() under _lock for stop_recording() resampling,
    # keeping open-and-record rate atomic without shared mutable state.
    return pa, stream, None, effective_rate


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


def refresh_pyaudio_session(
    pa_module: ModuleType,
    pyaudio_instance: "pyaudio.PyAudio | None",
    input_device_index: int | None,
    find_default_input_device: Callable[["pyaudio.PyAudio"], int],
) -> tuple["pyaudio.PyAudio | None", int | None]:
    """Device-change poll + on-demand construction (issue #42), called at
    the top of the recording worker before the retry loop.

    If ``pyaudio_instance`` is None (construction failed in __init__),
    construct now. Then poll the default input device against the cached
    instance: if the device moved (e.g. AirPods disconnected between
    recordings) or the index was never resolved, terminate the stale
    session and rebuild. A poll failure (enumeration OSError) is
    swallowed -- the retry loop still handles genuinely stale state,
    which is its original job (issue #37).

    Returns (pyaudio_instance, input_device_index) -- possibly both
    replaced. The caller is responsible for any lock scoping.
    """
    if pyaudio_instance is None:
        try:
            pyaudio_instance = pa_module.PyAudio()
        except Exception as e:  # noqa: BLE001 - PyAudio construction guard
            print(f"PyAudio construction failed at worker start: {e}")
        if pyaudio_instance is None:
            return None, None

    try:
        current_idx = pyaudio_instance.get_default_input_device_info()["index"]
    except (OSError, KeyError, TypeError):
        return pyaudio_instance, input_device_index

    if input_device_index is not None and current_idx == input_device_index:
        return pyaudio_instance, input_device_index  # device stable

    # Device moved (or index never resolved): rebuild the session against
    # the new default. A poll failure here (enumeration OSError, no input
    # device) falls through to the retry loop, which re-resolves fresh on
    # its own instances (issue #37).
    try:
        new_idx = find_default_input_device(pyaudio_instance)
    except (OSError, RuntimeError) as e:
        print(
            f"Default device poll failed at worker start: {e}; "
            "retry loop will re-resolve."
        )
        return pyaudio_instance, input_device_index

    if input_device_index is not None:
        # A prior device index exists: the stale session must go.
        # (Re-running the poll on a fresh instance could still show the
        # old default -- coreaudiod can lag a device switch -- in which
        # case the retry loop picks up the new device on its fresh
        # instances.)
        displaced = pyaudio_instance
        try:
            pyaudio_instance = pa_module.PyAudio()
        except Exception as e:  # noqa: BLE001 - PyAudio construction guard
            print(f"PyAudio construction failed at worker start: {e}")
            pyaudio_instance = displaced
            return pyaudio_instance, input_device_index
        terminate_quietly(displaced)

    return pyaudio_instance, new_idx
