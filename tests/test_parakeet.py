"""Tests for ParakeetTranscriber."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys


def _is_real_numpy(mod) -> bool:
    """True if `mod` is the real numpy module, not a MagicMock."""
    spec = getattr(mod, "__spec__", None)
    return spec is not None and getattr(spec, "name", None) == "numpy"


def _restore_real_numpy():
    """Pop mock numpy and ALL numpy.* submodules from sys.modules, then
    re-import the real numpy.

    Popping only `numpy` (without its submodules) causes a RecursionError
    on re-import: the import system finds `numpy.exceptions` already in
    sys.modules but its parent `numpy` is missing, so numpy's module-level
    `__getattr__` fires and re-triggers the partial-import cycle.
    """
    current = sys.modules.get("numpy")
    if current is not None and not _is_real_numpy(current):
        to_remove = [k for k in sys.modules if k == "numpy" or k.startswith("numpy.")]
        for k in to_remove:
            del sys.modules[k]
        import numpy as _real

        sys.modules["numpy"] = _real


# test_postprocessor.py installs a MagicMock numpy for the whole session;
# restore the real module if it was swapped out, so the local `np`
# binding is always the real one (issue #17 full-suite ordering fix).
_restore_real_numpy()
if not _is_real_numpy(np):
    np = sys.modules["numpy"]


def _ensure_real_numpy_in_module(module):
    """Restore real numpy in a module if an earlier test mocked it."""
    if not _is_real_numpy(module.np):
        module.np = np
        sys.modules["numpy"] = np


def make_mock_model():
    """Create a mock parakeet model."""
    mock_model = MagicMock()
    mock_model.preprocessor_config.sample_rate = 16000
    mock_result = MagicMock()
    mock_result.text = "hello world"
    mock_model.generate.return_value = [mock_result]
    return mock_model


# Mock hardware dependencies before importing kuiskaus modules
sys.modules["pyaudio"] = MagicMock()
sys.modules["mlx_whisper"] = MagicMock()
sys.modules["mlx_whisper.load_models"] = MagicMock()
sys.modules["parakeet_mlx"] = MagicMock()
sys.modules["parakeet_mlx.audio"] = MagicMock()
sys.modules["parakeet_mlx.audio"].get_logmel = MagicMock(return_value=MagicMock())
sys.modules["parakeet_mlx"].from_pretrained = MagicMock(return_value=make_mock_model())


class TestParakeetTranscriber:
    def setup_method(self):
        """Restore real numpy in the transcriber module before each test
        (test_postprocessor.py may have installed a MagicMock numpy for
        the whole session before this module was imported)."""
        import kuiskaus.parakeet_transcriber as pt

        _ensure_real_numpy_in_module(pt)

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
        from kuiskaus.transcriber import Transcriber
        from kuiskaus.parakeet_transcriber import ParakeetTranscriber

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
