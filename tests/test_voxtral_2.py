"""test_voxtral tests, part 2 of 2 (issue #70)."""

import os
import wave
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from tests.support_voxtral import (
    _EXC_REAL_BY_NAME,
)


class TestLoadErrorFormatting:
    """_format_load_error must distinguish Hugging Face availability
    failures from generic load failures (issue #30 DoD)."""

    @staticmethod
    def _resp(status_code: int) -> httpx.Response:
        return httpx.Response(
            status_code, request=httpx.Request("GET", "https://huggingface.co/api")
        )

    @staticmethod
    def _exc(
        cls_name: str,
        message: str,
        response: httpx.Response,
    ) -> HfHubHTTPError | GatedRepoError | RepositoryNotFoundError:
        """Build a real hf-hub exception instance for _format_load_error.

        The hf-hub stubs don't declare ``repo_id`` on the base HfHubHTTPError
        (only on the RepositoryNotFoundError subclass), so a direct attribute
        assignment on the base would trip the project's ty gate. We construct
        the real instance, set the stubs-invisible attribute via setattr
        (B010-suppressed), and the return-type union sidesteps the stub
        gap for downstream ty analysis (issue #30 review).
        """
        exc_cls = _EXC_REAL_BY_NAME[cls_name]
        exc = exc_cls(message, response=response)
        # B010-suppressed: the hf-hub stubs don't declare repo_id on the
        # base HfHubHTTPError, so direct assignment trips the ty gate.
        setattr(exc, "repo_id", "")  # noqa: B010
        return exc

    def test_repository_not_found(self):
        from kuiskaus.voxtral_transcriber import _format_load_error

        exc = self._exc("RepositoryNotFoundError", "msg", self._resp(404))
        setattr(exc, "repo_id", "mlx-community/does-not-exist")  # noqa: B010
        formatted = _format_load_error(exc)  # type: ignore[arg-type]
        assert "mlx-community/does-not-exist" in formatted
        assert "404" in formatted

    def test_gated_repo(self):
        from kuiskaus.voxtral_transcriber import _format_load_error

        exc = self._exc("GatedRepoError", "msg", self._resp(403))
        setattr(exc, "repo_id", "org/gated-model")  # noqa: B010
        formatted = _format_load_error(exc)  # type: ignore[arg-type]
        assert "org/gated-model" in formatted
        assert "auth" in formatted.lower()

    def test_gated_repo_falls_back_to_model_id_when_repo_id_unset(self):
        """HfHubHTTPError declares repo_id as str | None; when hf-hub can't
        parse it from the request URL the formatted cause must fall back
        to the configured model id, not render 'None'."""
        from kuiskaus.voxtral_transcriber import _MODEL_ID, _format_load_error

        exc = self._exc("GatedRepoError", "msg", self._resp(403))
        setattr(exc, "repo_id", None)  # noqa: B010
        formatted = _format_load_error(exc)  # type: ignore[arg-type]
        assert _MODEL_ID in formatted
        assert "None" not in formatted

    def test_hf_http_error_401(self):
        """A raw 401/403 (unauthenticated private repo) is an
        HfHubHTTPError without the not-found/gated refinements."""
        from kuiskaus.voxtral_transcriber import _format_load_error

        exc = self._exc(
            "HfHubHTTPError", "401 Client Error: Unauthorized", self._resp(401)
        )
        formatted = _format_load_error(exc)  # type: ignore[arg-type]
        assert "auth" in formatted.lower()
        assert "401" in formatted

    def test_offline_mode_error_is_surfaced_distinctly(self):
        """snapshot_download under HF_HUB_OFFLINE=1 raises
        OfflineModeIsEnabled; the formatted cause must name offline mode,
        not fall into the generic bucket (issue #30 review)."""
        from huggingface_hub.errors import OfflineModeIsEnabled

        from kuiskaus.voxtral_transcriber import _format_load_error

        formatted = _format_load_error(OfflineModeIsEnabled("offline mode is enabled"))
        assert "offline mode" in formatted.lower()
        assert "HF_HUB_OFFLINE" in formatted

    def test_generic_error_is_not_surfaced_verbatim(self):
        """Non-hf-hub exceptions must not leak raw third-party text into
        the user-facing string — only the exception class name."""
        from kuiskaus.voxtral_transcriber import _format_load_error

        message = (
            "Failed to download https://huggingface.co/x/weights.safetensors "
            "(server said: something opaque)"
        )
        assert (
            _format_load_error(RuntimeError(message))
            == "model load failed: RuntimeError"
        )
        assert "no load error recorded" in _format_load_error(None)

    def test_plain_error_with_availability_signature_maps_to_404(self):
        """A re-raised low-level failure (no hf-hub type on the object) whose
        message carries HF availability wording must still surface as the
        formatted 404, not the generic bucket (issue #30 DoD)."""
        from kuiskaus.voxtral_transcriber import _format_load_error

        exc = RuntimeError("404 Client Error: repository not found")
        assert "HTTP 404" in _format_load_error(exc)
        assert "not found" in _format_load_error(exc)
        assert "404 Client Error" not in _format_load_error(exc)

    def test_plain_error_with_unauthorized_signature_maps_to_auth(self):
        from kuiskaus.voxtral_transcriber import _format_load_error

        exc = RuntimeError("unauthorized for mzbac/voxtral-mini-3b-4bit-mixed")
        assert "auth" in _format_load_error(exc).lower()
        assert "401" in _format_load_error(exc)


class TestLoadErrorCaptureInLoadModel:
    """_load_model stores the first from_pretrained failure on
    _load_error and logs it; the stored cause is what _ensure_loaded
    surfaces."""

    def _transcriber_without_background_load(self):
        from kuiskaus.voxtral_transcriber import VoxtralTranscriber

        with patch("kuiskaus.voxtral_transcriber.VoxtralTranscriber._load_model"):
            return VoxtralTranscriber()

    def test_model_load_failure_stored_on_load_error(self):
        import mlx_voxtral as mv

        from kuiskaus.voxtral_transcriber import _MODEL_ID

        resp = httpx.Response(
            404, request=httpx.Request("GET", "https://huggingface.co/api")
        )

        exc = RepositoryNotFoundError("repository not found", response=resp)
        exc.repo_id = _MODEL_ID

        t = self._transcriber_without_background_load()
        with (
            patch("huggingface_hub.snapshot_download"),
            patch.object(
                mv.VoxtralForConditionalGeneration, "from_pretrained", side_effect=exc
            ),
            patch.object(mv.VoxtralProcessor, "from_pretrained") as mock_proc,
        ):
            t._load_model()
        assert t._model is None
        assert t._load_error is exc
        mock_proc.assert_not_called()

    def test_processor_load_failure_stored_on_load_error(self):
        import mlx_voxtral as mv

        from kuiskaus.voxtral_transcriber import _MODEL_ID

        t = self._transcriber_without_background_load()
        generic = OSError(
            f"{_MODEL_ID} is not a local folder or a valid repository name"
        )
        with (
            patch("huggingface_hub.snapshot_download"),
            patch.object(
                mv.VoxtralForConditionalGeneration, "from_pretrained"
            ) as mock_model,
            patch.object(mv.VoxtralProcessor, "from_pretrained", side_effect=generic),
        ):
            t._load_model()
        mock_model.assert_called_once()
        assert t._model is None
        assert t._load_error is generic

    def test_successful_load_clears_stale_load_error(self):
        """After a failed load, a successful reload on the same instance
        must reset _load_error so a stale failure can't surface as the
        current load's cause on a later cleanup() path."""
        import mlx_voxtral as mv

        from kuiskaus.voxtral_transcriber import _MODEL_ID

        resp = httpx.Response(
            404, request=httpx.Request("GET", "https://huggingface.co/api")
        )
        stale = RepositoryNotFoundError("repository not found", response=resp)
        stale.repo_id = _MODEL_ID

        t = self._transcriber_without_background_load()
        t._load_error = stale
        with (
            patch("huggingface_hub.snapshot_download"),
            patch.object(mv.VoxtralForConditionalGeneration, "from_pretrained"),
            patch.object(mv.VoxtralProcessor, "from_pretrained"),
        ):
            t._load_model()
        assert t._model is not None
        assert t._load_error is None

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
