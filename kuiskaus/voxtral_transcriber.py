"""Voxtral Realtime (Mistral) transcriber for Apple Silicon."""

import os
import tempfile
import threading
import time
import wave
from typing import TYPE_CHECKING, Any

import numpy as np

from .transcriber import TranscriptionResult

# mzbac/voxtral-mini-3b-4bit-mixed: public, unauthenticated, non-gated HF
# repo with the same VoxtralForConditionalGeneration architecture as the
# stock model, published as MLX-native quantized safetensors (~879 MB vs
# ~9.3 GB for mistralai's bf16 repo). The previous "mlx-community/..." id
# 404s on HF (issue #30).
_MODEL_ID = "mzbac/voxtral-mini-3b-4bit-mixed"

if TYPE_CHECKING:
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor


class VoxtralNotLoadedError(RuntimeError):
    """Raised when transcribe() is called before the model finished loading.

    Carries the original load failure in ``cause`` (``None`` if the load
    was never attempted, e.g. after cleanup() ran) and a pre-formatted,
    user-facing string in ``detail`` that distinguishes Hugging Face
    availability problems (repository not found / not authorised) from
    generic load failures.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        self.detail = f"{message} ({_format_load_error(cause)})"

    def __str__(self) -> str:
        # Surface the formatted cause, not just the bare message: this is
        # what the menubar prints at model-switch time (issue #30).
        return self.detail


def _format_load_error(error: Exception | None) -> str:
    """Format a load failure for surfacing to the user.

    Hugging Face availability failures are called out explicitly so the
    user can see "the model does not exist / you are not authorised".
    Everything else (OOM, network, corrupted weights) stays generic.
    """
    if error is None:
        return "no load error recorded"
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:  # pragma: no cover - mlx_voxtral requires hf-hub
        return str(error)
    if isinstance(error, GatedRepoError):
        return (
            f"'Hugging Face repository {error.repo_id} is gated or requires "
            "authentication; run `hf auth login` with a token that has access"
        )
    if isinstance(error, RepositoryNotFoundError):
        return f"Hugging Face repository '{error.repo_id}' not found (HTTP 404)"
    if isinstance(error, HfHubHTTPError):
        # Generic Hub HTTP error, e.g. 401/403 for a private repository.
        # The status code is embedded in the message hf-hub formats.
        return f"Hugging Face authentication or availability error: {error}"
    return str(error)


class VoxtralTranscriber:
    """Voxtral Realtime speech-to-text transcriber using mlx-voxtral."""

    def __init__(self) -> None:
        self._model: VoxtralForConditionalGeneration | None = None
        self._processor: VoxtralProcessor | None = None
        self._model_lock = threading.Lock()
        self._cleaned_up = False
        self._load_error: Exception | None = None
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    def _load_model(self) -> None:
        """Load Voxtral model and processor in background."""
        print(f"Loading Voxtral model: {_MODEL_ID}")
        start = time.time()
        try:
            from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

            # dtype must stay at from_pretrained's default: _MODEL_ID ships
            # quantized safetensors and mlx-voxtral skips dtype conversion
            # when config.json carries a "quantization" block.
            model = VoxtralForConditionalGeneration.from_pretrained(_MODEL_ID)
            processor = VoxtralProcessor.from_pretrained(_MODEL_ID)
            with self._model_lock:
                # Discard an in-flight load if cleanup() already ran: the
                # model would otherwise resurrect after being released.
                # Returning here unwinds the frame and drops the local
                # references, so nothing keeps the model resident.
                if self._cleaned_up:
                    return
                self._model = model
                self._processor = processor
            print(f"Voxtral model loaded in {time.time() - start:.2f}s")
        # Top-level guard for third-party model loading: must never let a
        # load failure crash the app; the error is logged here and
        # retained on self._load_error so _ensure_loaded() can surface
        # the original cause instead of a generic message (issue #30).
        except Exception as e:  # noqa: BLE001 - logged; model-load guard
            self._load_error = e
            print(f"Failed to load Voxtral model: {e}")

    @staticmethod
    def _input_ids(inputs: Any) -> Any:
        """Extract input_ids from processor output (attr or dict access)."""
        if isinstance(inputs, dict):
            return inputs["input_ids"]
        return inputs.input_ids

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
                "Voxtral model is not loaded",
                cause=self._load_error,
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
            with self._model_lock:
                start = time.time()
                inputs = processor.apply_transcrition_request(
                    language=kwargs.get("language", "en"),
                    audio=wav_path,
                )
                input_ids = self._input_ids(inputs)
                outputs = model.generate(
                    input_ids=input_ids,
                    input_features=inputs.get("input_features")
                    if isinstance(inputs, dict)
                    else inputs.input_features,
                    max_new_tokens=kwargs.get("max_new_tokens", 1024),
                    temperature=0.0,
                )
                text = processor.decode(
                    outputs[0][input_ids.shape[1] :],
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
        Note: the discarded load's model/processor references are held by
        the load thread until it returns, so peak memory is transiently
        ~2x the model during the load window after a mid-load cleanup.
        """
        with self._model_lock:
            self._cleaned_up = True
            # Drop the failure record too: after a release-and-reload
            # cycle a fresh load starts clean, and a stale 404 from a
            # previous load would otherwise surface on the next failed
            # load as if it were that load's own error.
            self._load_error = None
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
