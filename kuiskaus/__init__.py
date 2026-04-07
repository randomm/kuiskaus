"""
Kuiskaus - Whisper-powered speech-to-text for macOS
"""

__version__ = "1.0.0"
__author__ = "Kuiskaus Contributors"

from .audio_recorder import AudioRecorder
from .whisper_transcriber import WhisperTranscriber
from .transcriber import Transcriber, TranscriptionResult
from .text_inserter import TextInserter
from .hotkey_listener import HotkeyListener

__all__ = [
    "AudioRecorder",
    "WhisperTranscriber",
    "Transcriber",
    "TranscriptionResult",
    "TextInserter",
    "HotkeyListener",
]
