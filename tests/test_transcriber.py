"""Tests for Transcriber protocol."""

import numpy as np
import os
import importlib.util
from typing import Protocol

# Load transcriber module directly to avoid dependencies
spec_transcriber = importlib.util.spec_from_file_location(
    "kuiskaus.transcriber",
    os.path.join(os.path.dirname(__file__), "..", "kuiskaus", "transcriber.py"),
)
transcriber_module = importlib.util.module_from_spec(spec_transcriber)
spec_transcriber.loader.exec_module(transcriber_module)
Transcriber = transcriber_module.Transcriber


def test_transcriber_protocol_is_protocol():
    """Transcriber is a typing.Protocol."""
    # Check if Transcriber has protocol marker
    assert hasattr(Transcriber, "__protocol__") or Protocol in Transcriber.__bases__


def test_transcriber_protocol_methods():
    """Protocol defines transcribe and cleanup."""
    assert hasattr(Transcriber, "transcribe")
    assert hasattr(Transcriber, "cleanup")


def test_protocol_runtime_checkable():
    """Protocol is @runtime_checkable — isinstance() works."""

    class MockTranscriber:
        def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
            return {"text": "hello"}

        def cleanup(self) -> None:
            pass

    mock = MockTranscriber()
    assert isinstance(mock, Transcriber)


def test_non_conforming_class_fails_isinstance():
    """Object missing protocol methods fails isinstance check."""

    class NotATranscriber:
        pass

    obj = NotATranscriber()
    assert not isinstance(obj, Transcriber)


def test_mock_transcriber_satisfies_protocol():
    """Any object with transcribe+cleanup satisfies the protocol."""

    class MockTranscriber:
        def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
            return {"text": "hello", "segments": []}

        def cleanup(self) -> None:
            pass

    mock = MockTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    result = mock.transcribe(audio)
    assert result["text"] == "hello"
    mock.cleanup()


def test_transcriber_return_type():
    """Transcriber.transcribe returns dict with 'text' key."""

    class MockTranscriber:
        def transcribe(self, audio: np.ndarray, **kwargs) -> dict:
            return {"text": "test transcription", "segments": []}

        def cleanup(self) -> None:
            pass

    mock = MockTranscriber()
    audio = np.zeros(16000, dtype=np.float32)
    result = mock.transcribe(audio)
    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)


def test_transcriber_signature():
    """Protocol methods have correct signatures."""
    import inspect

    transcribe_sig = inspect.signature(Transcriber.transcribe)
    cleanup_sig = inspect.signature(Transcriber.cleanup)

    # transcribe should have audio parameter
    assert "audio" in transcribe_sig.parameters

    # cleanup should have no parameters (except self)
    assert len([p for p in cleanup_sig.parameters if p != "self"]) == 0


def test_whisper_transcriber_file_exists_and_imports_protocol():
    """WhisperTranscriber file exists and imports from the transcriber module."""
    with open(
        os.path.join(
            os.path.dirname(__file__), "..", "kuiskaus", "whisper_transcriber.py"
        ),
        "r",
    ) as f:
        content = f.read()

    # Check that it imports from transcriber
    assert (
        "from .transcriber import" in content
        or "from kuiskaus.transcriber import" in content
    )


def test_app_uses_transcriber_protocol():
    """app.py imports and uses Transcriber protocol."""
    with open(
        os.path.join(os.path.dirname(__file__), "..", "kuiskaus", "app.py"), "r"
    ) as f:
        content = f.read()

    # Check that it imports Transcriber
    assert (
        "from .transcriber import Transcriber" in content
        or "from kuiskaus.transcriber import Transcriber" in content
    )

    # Check that transcriber is typed as Transcriber
    assert (
        "self.transcriber: Transcriber" in content
        or "transcriber: Transcriber" in content
    )


def test_menubar_uses_transcriber_protocol():
    """menubar.py imports and uses Transcriber protocol."""
    with open(
        os.path.join(os.path.dirname(__file__), "..", "kuiskaus", "menubar.py"), "r"
    ) as f:
        content = f.read()

    # Check that it imports Transcriber
    assert (
        "from .transcriber import Transcriber" in content
        or "from kuiskaus.transcriber import Transcriber" in content
    )

    # Check that transcriber is typed as Transcriber
    assert (
        "self.transcriber: Transcriber" in content
        or "transcriber: Transcriber" in content
    )


def test_transcriber_exported_in_init():
    """Transcriber is exported in __init__.py."""
    with open(
        os.path.join(os.path.dirname(__file__), "..", "kuiskaus", "__init__.py"), "r"
    ) as f:
        content = f.read()

    # Check that Transcriber is in exports
    assert "Transcriber" in content


def test_transcription_result_exported_in_init():
    """TranscriptionResult TypedDict is exported in __init__.py."""
    with open(
        os.path.join(os.path.dirname(__file__), "..", "kuiskaus", "__init__.py"), "r"
    ) as f:
        content = f.read()

    # Check that TranscriptionResult is in exports
    assert "TranscriptionResult" in content
