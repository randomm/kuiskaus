"""Software resampling of captured audio to the transcription rate
(issue #55).

The recorder opens the stream at the device's native rate (issue #55)
to avoid PortAudio sample-rate renegotiation on macOS 26 Tahoe; this
module restores the 16 kHz mono float32 contract every downstream
transcriber (parakeet/whisper/voxtral) depends on, using numpy.interp
linear interpolation only.
"""

import numpy as np

#: The 16 kHz mono float32 contract all transcribers consume. The
#: transcribers import this constant for their duration math so the
#: rate cannot be desynchronized from the resampler.
TARGET_RATE = 16000


def resample_to_target(audio: np.ndarray, capture_rate: int | None) -> np.ndarray:
    """Resample ``audio`` (mono float32 at ``capture_rate`` Hz) to
    ``TARGET_RATE`` Hz.

    A capture rate of ``None`` (rate never recorded) or already equal
    to ``TARGET_RATE`` returns the input array unchanged (no-op,
    bit-identical -- no interpolation error introduced). Any other
    positive integer rate is resampled with linear interpolation to
    ``int(len(audio) * TARGET_RATE / capture_rate)`` samples.
    """
    if audio.size == 0:
        return audio
    if capture_rate is None or capture_rate == TARGET_RATE:
        return audio
    target_len = max(1, int(audio.size * TARGET_RATE / capture_rate))
    if target_len == audio.size:
        return audio
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
