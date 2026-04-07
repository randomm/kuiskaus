"""Transcriber Protocol for speech-to-text backends."""

from typing import Protocol, runtime_checkable, TypedDict
import numpy as np


class _TranscriptionBase(TypedDict):
    """Base TypedDict for transcription results with required fields."""

    text: str  # always required


class TranscriptionResult(_TranscriptionBase, total=False):
    """TypedDict for transcription results.

    The 'text' field is always required. Other fields are optional.
    """

    segments: list
    language: str
    transcribe_time: float
    audio_duration: float
    rtf: float


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for speech-to-text transcriber implementations.

    This Protocol is runtime checkable, so isinstance() can be used to verify
    that an implementation conforms to the interface.
    """

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz typically)
            **kwargs: Additional backend-specific parameters

        Returns:
            TranscriptionResult dict with at least 'text' key
        """
        ...

    def cleanup(self) -> None:
        """Release resources and cleanup."""
        ...
