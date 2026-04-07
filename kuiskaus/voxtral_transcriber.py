"""Voxtral Realtime (Mistral) transcriber for Apple Silicon."""

import tempfile
import time
import threading
import wave
import numpy as np
from typing import Optional, TYPE_CHECKING

from .transcriber import TranscriptionResult

_MODEL_ID = "mlx-community/Voxtral-Mini-3B-2507"

if TYPE_CHECKING:
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor


class VoxtralTranscriber:
    """Voxtral Realtime speech-to-text transcriber using mlx-voxtral."""

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self._model_id = model_id
        self._model: Optional["VoxtralForConditionalGeneration"] = None
        self._processor: Optional["VoxtralProcessor"] = None
        self._model_lock = threading.Lock()
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    def _load_model(self) -> None:
        """Load Voxtral model and processor in background."""
        print(f"Loading Voxtral model: {self._model_id}")
        start = time.time()
        try:
            from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

            with self._model_lock:
                self._model = VoxtralForConditionalGeneration.from_pretrained(
                    self._model_id
                )
                self._processor = VoxtralProcessor.from_pretrained(self._model_id)
            print(f"Voxtral model loaded in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"Failed to load Voxtral model: {e}")

    def _ensure_loaded(self) -> None:
        if self._load_thread.is_alive():
            print("Waiting for Voxtral model to load...")
            self._load_thread.join()
        if self._model is None or self._processor is None:
            raise RuntimeError("Voxtral model loading failed — check logs for errors")

    def _audio_to_wav_file(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Write numpy audio array to a temp WAV file. Returns file path."""
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
        except Exception:
            import os as _os

            _os.unlink(tmp_path)
            raise
        return tmp_path

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        """Transcribe audio array (float32, mono, 16kHz) using Voxtral."""
        if len(audio) == 0:
            return {"text": "", "language": "en"}

        self._ensure_loaded()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        import os

        text = ""
        transcribe_time = 0.0
        wav_path = self._audio_to_wav_file(audio)
        try:
            start = time.time()
            with self._model_lock:
                inputs = self._processor.apply_transcrition_request(
                    language=kwargs.get("language", "en"),
                    audio=wav_path,
                )
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 1024),
                    temperature=0.0,
                )
                text = self._processor.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
            transcribe_time = time.time() - start
        finally:
            os.unlink(wav_path)

        audio_duration = len(audio) / 16000.0
        return {
            "text": text,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
