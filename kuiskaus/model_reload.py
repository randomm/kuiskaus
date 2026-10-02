"""Background model switching for the menu bar app (issue #68).

Split out of menubar.py to keep it under the module size limit; the mixin
relies on the attributes the app sets in __init__.
"""

import threading
from typing import TYPE_CHECKING

from .parakeet_transcriber import ParakeetTranscriber
from .transcriber import Transcriber
from .whisper_transcriber import WhisperTranscriber

if TYPE_CHECKING:
    from .audio_recorder import AudioRecorder


class ModelReloadMixin:
    transcriber: Transcriber
    audio_recorder: "AudioRecorder"
    _transcriber_lock: threading.Lock
    _reload_lock: threading.Lock
    _reload_generation: int

    if TYPE_CHECKING:

        def update_status(self, status: str) -> None: ...

    def change_model(self, model_name: str):
        """Change the Whisper model"""
        self.update_status(f"Loading {model_name} model...")

        # Reload transcriber with new model
        threading.Thread(target=self._reload_model, args=(model_name,)).start()

    def _reload_model(self, model_name: str):
        """Reload the model in background.

        Reloads are serialized: each reload claims a generation under
        _reload_lock and, after loading, re-checks that it is still the
        latest. A superseded reload (one started while a newer reload is
        already running) must not commit its constructor result — that
        would roll back the newer reload's swap — and must not clean up
        the transcriber it found at start, which may already be live by
        then (issue #22). It DOES release its own (already loaded) model
        via best-effort cleanup() before discarding, so a superseded
        load never keeps a model resident.

        Success is reported only for a transcriber that is actually
        usable: Parakeet and Voxtral load in a background thread, so a
        successful constructor can still mean a failed (or still
        running) load; reporting the switch as done in that case would
        leave the UI claiming a working model while transcription cannot
        function (issue #22 review). An unusable result is discarded
        with the same failure path a constructor error takes.
        """
        try:
            with self._reload_lock:
                self._reload_generation += 1
                generation = self._reload_generation

            new_transcriber: Transcriber
            if model_name == "parakeet":
                new_transcriber = ParakeetTranscriber()
            elif model_name == "voxtral":
                from .voxtral_transcriber import VoxtralTranscriber

                new_transcriber = VoxtralTranscriber()
            else:
                new_transcriber = WhisperTranscriber(model_name=model_name)
            if not isinstance(new_transcriber, Transcriber):
                raise TypeError(
                    f"Transcriber implementation {type(new_transcriber)} does not satisfy "
                    "the Transcriber protocol"
                )
            # Background-load transcribers (Parakeet, Voxtral) swallow
            # load errors into their own state: the constructor succeeds
            # even when the model failed to load, so the reload must not
            # report success for a dead model (issue #22 review). Verify
            # against the REAL class (not the module attribute, which a
            # test patch may have replaced with a mock — isinstance() vs
            # a MagicMock raises TypeError): type() identity is stable
            # for stub instances whose class is a plain class.
            # Whisper loads eagerly in its constructor and raises on
            # failure, so its model is ready there.
            from .parakeet_transcriber import ParakeetTranscriber as _P
            from .voxtral_transcriber import VoxtralTranscriber as _V

            if type(new_transcriber) is _P:
                new_transcriber._ensure_loaded()
                if new_transcriber.model is None:
                    new_transcriber.cleanup()
                    raise RuntimeError("Model parakeet failed to load")
            elif type(new_transcriber) is _V:
                new_transcriber._ensure_loaded()
                if new_transcriber._model is None:
                    new_transcriber.cleanup()
                    raise RuntimeError("Model voxtral failed to load")

            with self._reload_lock:
                superseded = generation != self._reload_generation
            if superseded:
                # A newer reload is already in flight; don't commit this
                # (stale) result — it would roll back the newer reload's
                # swap and tear down a transcriber that may already be
                # live (issue #22). But the transcriber we just built IS
                # ours to release: its model is fully loaded here
                # (Parakeet/Whisper load in the constructor; for Voxtral
                # the load thread is a daemon we own and cleanup() stops
                # it), so release it before discarding.
                try:
                    new_transcriber.cleanup()
                except Exception as e:  # noqa: BLE001 - logged; best-effort release
                    print(f"⚠️  Superseded reload cleanup failed: {e}")
                print(f"⚠️  Model reload to {model_name} superseded; discarding")
                return

            # Read, swap, then clean up the old transcriber: the worker
            # holds _transcriber_lock for its entire transcribe() call, so
            # the snapshot-and-swap isolates an in-flight worker from the
            # teardown (it keeps its own reference and finishes on a live
            # object). Cleanup of the old model can take seconds, so it
            # runs outside the lock (issue #22). Best-effort guard, same
            # as the superseded-reload branch above: a failing cleanup
            # must not mask a successful swap (issue #22 review).
            with self._transcriber_lock:
                old_transcriber = self.transcriber
                self.transcriber = new_transcriber
            try:
                old_transcriber.cleanup()
            except Exception as e:  # noqa: BLE001 - logged; best-effort release
                print(f"⚠️  Old transcriber cleanup failed: {e}")

            # A model switch never touches the microphone, so it must not
            # clear a live mic-error banner (#16 DoD: persists until the
            # next successful *recording*, no other auto-clear trigger).
            if not self.audio_recorder.last_error:
                self.update_status("🟢 Ready")
            print(f"✅ Model changed to {model_name}")
        # Top-level guard for the model-reload worker thread: model loading
        # can fail for many reasons and must not crash the app.
        except Exception as e:  # noqa: BLE001 - logged; worker-thread guard
            # Same rationale as the success path above.
            if not self.audio_recorder.last_error:
                self.update_status("🟢 Ready (model error)")
            print(f"❌ Model error: {e}")
