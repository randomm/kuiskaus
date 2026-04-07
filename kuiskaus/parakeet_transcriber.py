"""Parakeet TDT 0.6B v3 transcriber for Apple Silicon."""

import time
import threading
import numpy as np

from .transcriber import TranscriptionResult


class ParakeetTranscriber:
    """Parakeet TDT 0.6B v3 speech-to-text transcriber using MLX."""

    MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self) -> None:
        self.model = None
        self._model_lock = threading.Lock()
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    def _load_model(self) -> None:
        """Load Parakeet model in background thread."""
        print(f"Loading Parakeet model: {self.MODEL_ID}")
        start = time.time()
        try:
            from parakeet_mlx import from_pretrained

            with self._model_lock:
                self.model = from_pretrained(self.MODEL_ID)
            print(f"Parakeet model loaded in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"Failed to load Parakeet model: {e}")
            # model stays None — RuntimeError will be raised on next transcribe()

    def _ensure_loaded(self) -> None:
        if self._load_thread.is_alive():
            print("Waiting for Parakeet model to load...")
            self._load_thread.join()

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        """Transcribe audio array (float32, mono, 16kHz) to text."""
        self._ensure_loaded()
        if self.model is None:
            raise RuntimeError("Parakeet model failed to load")

        if len(audio) == 0:
            return {
                "text": "",
                "transcribe_time": 0.0,
                "audio_duration": 0.0,
                "rtf": 0.0,
            }

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start = time.time()

        from parakeet_mlx.audio import get_logmel
        import mlx.core as mx

        with self._model_lock:
            mel = get_logmel(mx.array(audio), self.model.preprocessor_config)
            alignments = self.model.generate(mel)

        transcribe_time = time.time() - start
        audio_duration = len(audio) / 16000.0

        text = ""
        if alignments:
            text = alignments[0].text.strip()

        return {
            "text": text,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        with self._model_lock:
            if self.model is not None:
                del self.model
                self.model = None
