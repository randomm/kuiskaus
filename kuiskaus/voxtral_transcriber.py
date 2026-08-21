"""Voxtral Realtime (Mistral) transcriber for Apple Silicon."""

import os
import tempfile
import threading
import time
import wave
from typing import TYPE_CHECKING

import numpy as np

from .transcriber import TranscriptionResult

_MODEL_ID = "mlx-community/Voxtral-Mini-3B-2507"

if TYPE_CHECKING:
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor


class VoxtralNotLoadedError(RuntimeError):
    """Raised when transcribe() is called before the model finished loading."""


class VoxtralTranscriber:
    """Voxtral Realtime speech-to-text transcriber using mlx-voxtral."""

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self._model_id = model_id
        self._model: VoxtralForConditionalGeneration | None = None
        self._processor: VoxtralProcessor | None = None
        self._model_lock = threading.Lock()
        self._cleaned_up = False
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    def _load_model(self) -> None:
        """Load Voxtral model and processor in background."""
        print(f"Loading Voxtral model: {self._model_id}")
        start = time.time()
        try:
            from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

            model = VoxtralForConditionalGeneration.from_pretrained(self._model_id)
            processor = VoxtralProcessor.from_pretrained(self._model_id)
            with self._model_lock:
                # Discard an in-flight load if cleanup() already ran: the
                # model would otherwise resurrect after being released.
                if self._cleaned_up:
                    return
                self._model = model
                self._processor = processor
            print(f"Voxtral model loaded in {time.time() - start:.2f}s")
        # Top-level guard for third-party model loading: must never let a
        # load failure crash the app; the error is logged here.
        except Exception as e:  # noqa: BLE001 - logged; model-load guard
            print(f"Failed to load Voxtral model: {e}")

    def _ensure_loaded(
        self,
    ) -> tuple["VoxtralForConditionalGeneration", "VoxtralProcessor"]:
        """Wait for the background load and return the loaded model/processor.

        Raises VoxtralNotLoadedError if the load never succeeded, so callers
        cannot dereference ``self._model`` / ``self._processor`` while they
        are still ``None`` (or were reset by cleanup()).
        """
        if self._load_thread.is_alive():
            print("Waiting for Voxtral model to load...")
            self._load_thread.join()
        if self._model is None or self._processor is None:
            raise VoxtralNotLoadedError(
                "Voxtral model is not loaded — loading failed or cleanup() ran; "
                "check logs for the original error"
            )
        return self._model, self._processor

    def _audio_to_wav_file(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Write numpy audio array to a temp WAV file. Returns file path."""
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        # NamedTemporaryFile(delete=False) is intentional: the path must
        # outlive the context block so the model can read the WAV file
        # later; the caller unlinks it in a finally block.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
        except BaseException:
            # Unlink on ANY failure class (wave.Error and ValueError are not
            # OSError), then re-raise untouched.
            os.unlink(tmp_path)
            raise
        return tmp_path

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        """Transcribe audio array (float32, mono, 16kHz) using Voxtral."""
        if len(audio) == 0:
            return {"text": "", "language": "en"}

        model, processor = self._ensure_loaded()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        text = ""
        transcribe_time = 0.0
        wav_path: str | None = None
        try:
            wav_path = self._audio_to_wav_file(audio)
            start = time.time()
            with self._model_lock:
                inputs = processor.apply_transcrition_request(
                    language=kwargs.get("language", "en"),
                    audio=wav_path,
                )
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 1024),
                    temperature=0.0,
                )
                text = processor.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
            transcribe_time = time.time() - start
        finally:
            if wav_path is not None:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        audio_duration = len(audio) / 16000.0
        return {
            "text": text,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }

    def cleanup(self) -> None:
        """Release model resources.

        Sticks: once cleanup() has run, an in-flight background load
        discards its result instead of repopulating the released model.
        """
        with self._model_lock:
            self._cleaned_up = True
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
