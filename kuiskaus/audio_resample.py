"""Software resampling of captured audio to the transcription rate
(issue #55).

The recorder opens the stream at the device's native rate (issue #55)
to avoid PortAudio sample-rate renegotiation on macOS 26 Tahoe; this
module restores the 16 kHz mono float32 contract every downstream
transcriber (parakeet/whisper/voxtral) depends on, using index-based
linear interpolation (no scipy/librosa dependency).
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
    if capture_rate <= 0:
        raise ValueError(f"capture_rate must be positive, got {capture_rate}")
    target_len = max(1, int(audio.size * TARGET_RATE / capture_rate))
    if target_len == audio.size:
        return audio
    # Index-based linear interpolation: this runs ONCE on the fully
    # assembled buffer in stop_recording() (not per-chunk), so the
    # allocation cost scales with the whole recording (a 60 s capture at
    # 48 kHz native is ~2.88M samples). Interpolating via integer indices
    # into the source allocates O(target_len) working space instead of
    # np.interp's O(audio.size) linspace + O(target_len) float64
    # temporaries (issue #55 lens review MEDIUM, performance).
    idx = np.arange(target_len, dtype=np.float64) * (audio.size / target_len)
    lo = np.clip(idx.astype(np.intp), 0, audio.size - 1)
    hi = np.clip(lo + 1, 0, audio.size - 1)
    frac = (idx - lo).astype(np.float32)
    result = audio[lo] * (1.0 - frac) + audio[hi] * frac
    return result.astype(np.float32)
