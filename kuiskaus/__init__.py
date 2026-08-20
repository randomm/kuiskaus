"""
Kuiskaus - Whisper-powered speech-to-text for macOS
"""

__version__ = "1.0.0"
__author__ = "Kuiskaus Contributors"

from .audio_recorder import AudioRecorder
from .hotkey_listener import HotkeyListener
from .parakeet_transcriber import ParakeetTranscriber
from .text_inserter import TextInserter
from .transcriber import Transcriber, TranscriptionResult
from .voxtral_transcriber import VoxtralTranscriber
from .whisper_transcriber import WhisperTranscriber

__all__ = [
    "AudioRecorder",
    "HotkeyListener",
    "ParakeetTranscriber",
    "TextInserter",
    "Transcriber",
    "TranscriptionResult",
    "VoxtralTranscriber",
    "WhisperTranscriber",
]
