"""Tests for VoxtralTranscriber."""

import os
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# NOTE: No module-level sys.modules stubbing here. The hardware deps
# (pyaudio, mlx_voxtral, parakeet_mlx) are real on Apple Silicon and are
# imported lazily inside methods; tests patch at the method level so
# nothing leaks into the shared pytest process.


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

    def test_audio_to_wav_unlinks_on_non_oserror_failure(self):
        """A wave.Error (not OSError) during write must still unlink the temp file."""
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
        t._load_thread.join(timeout=1)
        audio = np.zeros(16000, dtype=np.float32)
        with (
            patch("wave.open") as mock_open,
            patch("os.unlink") as mock_unlink,
        ):
            mock_wf = MagicMock()
            mock_wf.setframerate.side_effect = wave.Error("bad frame rate")
            mock_open.return_value.__enter__.return_value = mock_wf
            with pytest.raises(wave.Error):
                t._audio_to_wav_file(audio, sample_rate=0)
        # Unlinked exactly once, with the created temp path
        assert len(mock_unlink.call_args_list) == 1
        unlinked_path = mock_unlink.call_args_list[0].args[0]
        assert unlinked_path.endswith(".wav")

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

    def test_cleanup_sticks_when_load_in_flight(self):
        """cleanup() during an in-flight load must not be undone by the load.

        Reproduction of the resurrection-after-cleanup race: cleanup() runs
        while _load_model is still between "model loaded" and "model
        stored"; without the _cleaned_up guard the load repopulates
        _model/_processor afterwards.
        """
        # Warm the mlx_voxtral import on the main thread BEFORE starting
        # the load thread: on a cold CI runner the first import of
        # mlx_voxtral (which pulls in torch/transformers) takes far
        # longer than the 5s gate budget, so leaving it for the thread
        # made "load thread did not reach the gate" race against the
        # import time (issue #22, CI run 32517118857). The thread's
        # job is only to reach the patched from_pretrained, so the
        # import cost must sit outside the timed window.
        import mlx_voxtral  # noqa: F401  # warm-up; see note above

        from kuiskaus.voxtral_transcriber import _MODEL_ID, VoxtralTranscriber

        # The load thread blocks on `release` until the main thread has
        # called cleanup() and set it — the exact in-flight window of the
        # race. We use a plain bool + Event so the load thread can't
        # accidentally see the flag as already-set.
        release = threading.Event()

        # load_model_gate: the first from_pretrained call (the model).
        # load_processor_gate: the second from_pretrained call (the
        # processor) — where the load is in flight, so the gate blocks
        # there to hold the race window open.
        def load_model_gate(_model_id: str) -> MagicMock:
            assert _model_id == _MODEL_ID
            return MagicMock()

        def load_processor_gate(_model_id: str) -> MagicMock:
            # Signal that the load is in flight, then block until released
            gate.set()
            release.wait(timeout=5)
            assert _model_id == _MODEL_ID
            return MagicMock()

        gate = threading.Event()

        # Suppress the auto-started background load so we can run
        # _load_model deterministically on our own thread.
        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()

        def load_with_gate():
            # The mlx_voxtral import was already paid on the main thread
            # above; from here on it is a sys.modules cache hit.
            # patch.object targets the real class objects: _load_model
            # does `from mlx_voxtral import ...`, which rebinds to the
            # same real objects, so the attribute patches are visible
            # inside it.
            import mlx_voxtral as mv

            with (
                patch.object(
                    mv.VoxtralForConditionalGeneration,
                    "from_pretrained",
                    side_effect=load_model_gate,
                ),
                patch.object(
                    mv.VoxtralProcessor,
                    "from_pretrained",
                    side_effect=load_processor_gate,
                ),
            ):
                t._load_model()

        thread = threading.Thread(target=load_with_gate, daemon=True)
        thread.start()
        assert gate.wait(5), "load thread did not reach the gate"

        t.cleanup()
        release.set()  # release the load; it now stores (or discards) the model
        thread.join(timeout=5)
        assert not thread.is_alive()

        assert t._model is None, "model resurrected after cleanup()"
        assert t._processor is None, "processor resurrected after cleanup()"

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


def test_no_sys_modules_pollution_after_import():
    """Importing and running this file must not leak MagicMock stubs into
    sys.modules: a later import of a real dependency must see the real
    module, not a MagicMock. A real module has a __file__ path; a
    MagicMock injected into sys.modules does not."""
    import mlx_voxtral

    assert not isinstance(mlx_voxtral, MagicMock)
    assert getattr(mlx_voxtral, "__file__", None) is not None
