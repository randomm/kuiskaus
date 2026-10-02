"""test_voxtral tests, part 1 of 2 (issue #70)."""

import os
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestVoxtralTranscriber:
    def _make_transcriber_with_mock(self):
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            t = VoxtralTranscriber()
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.decode.return_value = "hello voxtral"
        # mlx_voxtral's apply_transcrition_request returns a
        # TranscriptionInputs object (attributes, not a dict); the mock
        # mirrors that contract.
        mock_processor.apply_transcrition_request.return_value = SimpleNamespace(
            input_ids=MagicMock(shape=(1, 10)),
            input_features=MagicMock(),
        )
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

        from kuiskaus.voxtral_transcriber import (
            _MODEL_ID,
            _MODEL_REVISION,
            VoxtralTranscriber,
        )

        # The load thread blocks on `release` until the main thread has
        # called cleanup() and set it — the exact in-flight window of the
        # race. We use a plain bool + Event so the load thread can't
        # accidentally see the flag as already-set.
        release = threading.Event()

        # load_model_gate: the first from_pretrained call (the model).
        # load_processor_gate: the second from_pretrained call (the
        # processor) — where the load is in flight, so the gate blocks
        # there to hold the race window open.
        # Both from_pretrained calls receive the pinned snapshot path
        # (offline-safe; see _load_model), so the gates just accept it.
        def load_model_gate(model_path: str) -> MagicMock:
            return MagicMock()

        def load_processor_gate(model_path: str) -> MagicMock:
            # Signal that the load is in flight, then block until released
            gate.set()
            release.wait(timeout=5)
            assert model_path == str(Path("/tmp/pinned-snapshot"))
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
                patch(
                    "huggingface_hub.snapshot_download",
                    return_value=Path("/tmp/pinned-snapshot"),
                ) as mock_snapshot,
            ):
                t._load_model()
                # The pin must be enforced: snapshot_download resolves the
                # pinned revision, and the model/processor load from that
                # snapshot path.
                mock_snapshot.assert_called_once_with(
                    _MODEL_ID, revision=_MODEL_REVISION
                )

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
        with pytest.raises(RuntimeError, match="not loaded"):
            t.transcribe(audio)

    def test_load_failure_records_cause_on_not_loaded_error(self):
        """A failed load must surface the original exception, not just a
        generic 'loading failed' message (issue #30 DoD: a 404 should not
        look the same as an out-of-memory failure)."""
        from kuiskaus.voxtral_transcriber import (
            VoxtralNotLoadedError,
            VoxtralTranscriber,
        )

        with patch.object(VoxtralTranscriber, "_load_model"):
            t = VoxtralTranscriber()
        t._load_thread.join(timeout=1)
        load_error = RuntimeError("404 Client Error: repository not found")
        t._load_error = load_error
        t._model = None
        t._processor = None
        with pytest.raises(VoxtralNotLoadedError) as exc_info:
            t.transcribe(np.zeros(16000, dtype=np.float32))
        assert exc_info.value.cause is load_error
        # The cause is attached, but the formatted string must not leak the
        # raw third-party message verbatim (only the exception class name).
        assert "404 Client Error: repository not found" not in str(exc_info.value)
        # A plain exception whose message carries an HF availability
        # signature (e.g. a re-raised 404) surfaces the formatted 404
        # cause, not the generic bucket (issue #30 DoD).
        assert "HTTP 404" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_cleanup_drops_recorded_load_error(self):
        """cleanup() resets _load_error so a stale failure cannot surface
        as a later load's error on the reload path."""
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch.object(VoxtralTranscriber, "_load_model"):
            t = VoxtralTranscriber()
        t._load_thread.join(timeout=1)
        t._load_error = RuntimeError("stale 404")
        t.cleanup()
        assert t._load_error is None
