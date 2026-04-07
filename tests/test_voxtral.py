"""Tests for VoxtralTranscriber."""

import numpy as np
import os
import sys
import wave
import pytest
from unittest.mock import MagicMock, patch

sys.modules["pyaudio"] = MagicMock()
sys.modules["mlx_whisper"] = MagicMock()
sys.modules["mlx_whisper.load_models"] = MagicMock()
sys.modules["mlx_voxtral"] = MagicMock()
sys.modules["parakeet_mlx"] = MagicMock()
sys.modules["parakeet_mlx.audio"] = MagicMock()


class TestVoxtralTranscriber:
    def _make_transcriber_with_mock(self):
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.decode.return_value = "hello voxtral"
        mock_processor.apply_transcrition_request.return_value = {
            "input_ids": MagicMock(shape=(1, 10))
        }
        mock_model.generate.return_value = [MagicMock()]
        t._model = mock_model
        t._processor = mock_processor
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
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
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
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
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
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
        assert isinstance(t, Transcriber)

    def test_cleanup_releases_model(self):
        t = self._make_transcriber_with_mock()
        t.cleanup()
        assert t._model is None
        assert t._processor is None

    def test_transcribe_raises_if_model_not_loaded(self):
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
        t._model = None
        t._processor = None
        t._load_thread.join(timeout=1)
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="loading failed"):
            t.transcribe(audio)

    def test_audio_clipping(self):
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
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
