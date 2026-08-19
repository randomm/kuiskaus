"""Tests for VoxtralTranscriber."""

import numpy as np
import os
import sys
import wave
import pytest
from unittest.mock import MagicMock, patch


def _is_real_numpy(mod) -> bool:
    """True if `mod` is the real numpy module, not a MagicMock.

    MagicMock auto-creates attributes (so `hasattr(mock, "float32")` is
    always True) but has no real `__spec__`; the real numpy module does.
    """
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

sys.modules["pyaudio"] = MagicMock()
sys.modules["mlx_whisper"] = MagicMock()
sys.modules["mlx_whisper.load_models"] = MagicMock()
sys.modules["mlx_voxtral"] = MagicMock()
sys.modules["parakeet_mlx"] = MagicMock()
sys.modules["parakeet_mlx.audio"] = MagicMock()


def _ensure_real_numpy_in_module(module):
    """If an earlier test module mocked numpy before `module` was imported,
    its module-level `np` binding may be a MagicMock. Restore the real
    module so its numpy calls work (this file's `np` is real)."""
    if not _is_real_numpy(module.np):
        module.np = np
        # Also restore the real numpy in the session so any code that
        # re-imports it gets the real module.
        if not _is_real_numpy(sys.modules.get("numpy")):
            _restore_real_numpy()


class TestVoxtralTranscriber:
    def setup_method(self):
        """Restore real numpy in the transcriber module before each test
        (test_postprocessor.py may have installed a MagicMock numpy for
        the whole session before this module was imported)."""
        import kuiskaus.voxtral_transcriber as vt

        _ensure_real_numpy_in_module(vt)

    def _make_transcriber_with_mock(self):
        import kuiskaus.voxtral_transcriber as vt

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = vt.VoxtralTranscriber()
        t._model = MagicMock()
        t._processor = MagicMock()
        t._processor.decode.return_value = "hello voxtral"
        t._processor.apply_transcrition_request.return_value = {
            "input_ids": MagicMock(shape=(1, 10))
        }
        t._load_thread.join(timeout=1)
        return t

    def test_transcribe_returns_text(self):
        t = self._make_transcriber_with_mock()
        audio = np.zeros(16000, dtype=np.float32)
        with (
            patch.object(t, "_audio_to_wav_file", return_value="/tmp/fake.wav"),
            patch("os.unlink"),
        ):
            result = t.transcribe(audio)
        assert result["text"] == "hello voxtral"

    def test_transcribe_empty_returns_empty(self):
        import kuiskaus.voxtral_transcriber as vt

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = vt.VoxtralTranscriber()
        t._load_thread.join(timeout=1)
        result = t.transcribe(np.array([], dtype=np.float32))
        assert result["text"] == ""

    def test_transcribe_includes_timing(self):
        t = self._make_transcriber_with_mock()
        audio = np.zeros(16000, dtype=np.float32)
        with (
            patch.object(t, "_audio_to_wav_file", return_value="/tmp/fake.wav"),
            patch("os.unlink"),
        ):
            result = t.transcribe(audio)
        assert "transcribe_time" in result
        assert "audio_duration" in result
        assert "rtf" in result

    def test_wav_file_cleaned_up(self):
        t = self._make_transcriber_with_mock()
        audio = np.zeros(16000, dtype=np.float32)
        with (
            patch.object(t, "_audio_to_wav_file", return_value="/tmp/test.wav"),
            patch("os.unlink") as mock_unlink,
        ):
            t.transcribe(audio)
        mock_unlink.assert_called_once_with("/tmp/test.wav")

    def test_audio_to_wav_creates_valid_wav(self):
        import kuiskaus.voxtral_transcriber as vt

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = vt.VoxtralTranscriber()
        t._load_thread.join(timeout=1)
        audio = np.zeros(16000, dtype=np.float32)
        path = t._audio_to_wav_file(audio)
        try:
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 16000
        finally:
            os.unlink(path)

    def test_transcribe_satisfies_protocol(self):
        from kuiskaus.transcriber import Transcriber
        import kuiskaus.voxtral_transcriber as vt

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = vt.VoxtralTranscriber()
        assert isinstance(t, Transcriber)

    def test_cleanup_releases_model(self):
        t = self._make_transcriber_with_mock()
        t.cleanup()
        assert t._model is None
        assert t._processor is None

    def test_transcribe_raises_if_model_not_loaded(self):
        import kuiskaus.voxtral_transcriber as vt

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = vt.VoxtralTranscriber()
        t._model = None
        t._processor = None
        t._load_thread.join(timeout=1)
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="loading failed"):
            t.transcribe(audio)

    def test_audio_clipping(self):
        import kuiskaus.voxtral_transcriber as vt

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = vt.VoxtralTranscriber()
        t._load_thread.join(timeout=1)
        audio = np.array([1.5, -1.5, 0.5], dtype=np.float32)
        path = t._audio_to_wav_file(audio)
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.readframes(3)
            samples = np.frombuffer(frames, dtype=np.int16)
            assert samples[0] == 32767
            assert samples[1] == -32768
        finally:
            os.unlink(path)
