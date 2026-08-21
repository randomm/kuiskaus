"""Tests for ParakeetTranscriber.

No module-level sys.modules stubbing: parakeet_mlx is a real dependency on
Apple Silicon and is imported lazily inside methods; tests patch at the
method level so nothing leaks into the shared pytest process.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def make_mock_model():
    """Create a mock parakeet model."""
    mock_model = MagicMock()
    mock_model.preprocessor_config.sample_rate = 16000
    mock_result = MagicMock()
    mock_result.text = "hello world"
    mock_model.generate.return_value = [mock_result]
    return mock_model


class TestParakeetTranscriber:
    def _make_transcriber_with_mock(self):
        """Create a ParakeetTranscriber with mocked model (skips loading)."""
        from kuiskaus.parakeet_transcriber import ParakeetTranscriber

        with patch("kuiskaus.parakeet_transcriber.ParakeetTranscriber._load_model"):
            transcriber = ParakeetTranscriber()
        transcriber.model = make_mock_model()
        # Ensure load thread is done
        transcriber._load_thread.join(timeout=1)
        return transcriber

    def test_transcribe_returns_text(self):
        t = self._make_transcriber_with_mock()
        audio = np.zeros(16000, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result["text"] == "hello world"

    def test_transcribe_empty_audio_returns_empty(self):
        from kuiskaus.parakeet_transcriber import ParakeetTranscriber

        with patch("kuiskaus.parakeet_transcriber.ParakeetTranscriber._load_model"):
            t = ParakeetTranscriber()
        t.model = make_mock_model()
        t._load_thread.join(timeout=1)
        result = t.transcribe(np.array([], dtype=np.float32))
        assert result["text"] == ""
        assert result["transcribe_time"] == 0.0
        assert result["audio_duration"] == 0.0
        assert result["rtf"] == 0.0

    def test_transcribe_converts_dtype(self):
        t = self._make_transcriber_with_mock()
        audio = np.zeros(16000, dtype=np.int16)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert "text" in result

    def test_transcribe_includes_timing(self):
        t = self._make_transcriber_with_mock()
        audio = np.zeros(16000, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert "transcribe_time" in result
        assert "audio_duration" in result
        assert "rtf" in result

    def test_transcribe_satisfies_protocol(self):
        from kuiskaus.parakeet_transcriber import ParakeetTranscriber
        from kuiskaus.transcriber import Transcriber

        with patch("kuiskaus.parakeet_transcriber.ParakeetTranscriber._load_model"):
            t = ParakeetTranscriber()
        assert isinstance(t, Transcriber)

    def test_cleanup_releases_model(self):
        t = self._make_transcriber_with_mock()
        t.cleanup()
        assert t.model is None

    def test_transcribe_raises_if_model_not_loaded(self):
        from kuiskaus.parakeet_transcriber import ParakeetTranscriber

        with patch("kuiskaus.parakeet_transcriber.ParakeetTranscriber._load_model"):
            t = ParakeetTranscriber()
        t.model = None  # simulate failed load
        t._load_thread.join(timeout=1)
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="model failed to load"):
            t.transcribe(audio)


def test_no_sys_modules_pollution_after_import():
    """Importing and running this file must not leak MagicMock stubs into
    sys.modules: a later import of a real dependency must see the real
    module, not a MagicMock. A real module has a __file__ path; a
    MagicMock injected into sys.modules does not."""
    import parakeet_mlx

    assert not isinstance(parakeet_mlx, MagicMock)
    assert getattr(parakeet_mlx, "__file__", None) is not None
