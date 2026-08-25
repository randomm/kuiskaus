#!/usr/bin/env python3
"""
Kuiskaus Menu Bar App - Whisper V3 Turbo Speech-to-Text for macOS
A menu bar application for easy access to speech-to-text functionality.
"""

import queue
import threading
import time
from datetime import UTC, datetime

import numpy as np
import rumps

from .audio_recorder import AudioRecorder
from .hotkey_listener_cgevent import HotkeyListenerCGEvent
from .parakeet_transcriber import ParakeetTranscriber
from .postprocessor import clean_with_apfel
from .silicon_check import check_apple_silicon
from .text_inserter import TextInserter
from .transcriber import Transcriber
from .whisper_transcriber import WhisperTranscriber


def _utcnow() -> datetime:
    """Single source of aware-UTC now(); prevents naive-datetime subtraction errors."""
    return datetime.now(tz=UTC)


class KuiskausMenuBarApp(rumps.App):
    def __init__(self):
        super().__init__(
            "Kuiskaus",
            title="🎤",  # Use title instead of icon for emoji
            quit_button=None,  # We'll add custom quit with stats
        )

        # Initialize components
        self.audio_recorder = AudioRecorder(
            on_capture_started=self._enqueue_capture_started
        )
        self.transcriber: Transcriber = ParakeetTranscriber()
        if not isinstance(self.transcriber, Transcriber):
            raise TypeError(
                f"Transcriber implementation {type(self.transcriber)} does not satisfy "
                "the Transcriber protocol"
            )
        self.text_inserter = TextInserter()

        # State
        self.is_recording = False
        self.recording_start_time = None
        self.enabled = True
        self.use_apfel: bool = False
        self._apfel_lock = threading.Lock()
        # Guards self.transcriber: held across transcribe() calls; see
        # _reload_model for the full rationale.
        self._transcriber_lock = threading.Lock()
        # Serializes reload threads: a superseded reload discards its
        # constructor result instead of committing or tearing down a
        # transcriber that a newer reload already swapped in (issue #22).
        self._reload_lock = threading.Lock()
        self._reload_generation = 0
        # Worker-thread -> main-thread trampoline for the capture-started
        # event (issue #43): the recorder's worker calls
        # _enqueue_capture_started off the main thread; the rumps.Timer
        # below drains the queue on the main thread, where UI mutation is
        # safe. rumps is main-thread-affine, so direct writes from the
        # worker would race the NSRunLoop.
        self._pending_capture_started_events: queue.Queue[int] = queue.Queue()
        self._ui_tick_timer = rumps.Timer(self._drain_ui_events, 0.05)
        self._ui_tick_timer.start()

        # Initialize hotkey listener with CGEventTap
        self.hotkey_listener = HotkeyListenerCGEvent(
            on_press=self.on_hotkey_press, on_release=self.on_hotkey_release
        )

        # Stats
        self.total_transcriptions = 0
        self.total_recording_time = 0.0
        self.session_start = _utcnow()

        # Setup menu
        self.setup_menu()

        # Start hotkey listener in background
        self.start_hotkey_listener()

    def setup_menu(self):
        """Setup the menu bar items"""
        # Status item (will be updated dynamically)
        self.status_item = rumps.MenuItem("🟢 Ready", callback=None)
        self.menu.add(self.status_item)
        self.menu.add(rumps.separator)

        # Enable/Disable toggle
        self.enable_item = rumps.MenuItem("Enabled", callback=self.toggle_enabled)
        self.enable_item.state = True
        self.menu.add(self.enable_item)

        self.apfel_item = rumps.MenuItem(
            "LLM Cleanup (apfel)", callback=self.toggle_apfel
        )
        self.apfel_item.state = False
        self.menu.add(self.apfel_item)

        # Hotkey info
        self.menu.add(rumps.MenuItem("Hotkey: ⌃⌥ (Control+Option)", callback=None))
        self.menu.add(rumps.separator)

        # Model selection submenu
        model_menu = rumps.MenuItem("Model")
        model_menu.add(
            rumps.MenuItem(
                "Parakeet TDT 0.6B v3 (Default)",
                callback=lambda _: self.change_model("parakeet"),
            )
        )
        model_menu.add(
            rumps.MenuItem(
                "Voxtral Realtime",
                callback=lambda _: self.change_model("voxtral"),
            )
        )
        model_menu.add(
            rumps.MenuItem(
                "Whisper Turbo", callback=lambda _: self.change_model("turbo")
            )
        )
        model_menu.add(
            rumps.MenuItem("Base", callback=lambda _: self.change_model("base"))
        )
        model_menu.add(
            rumps.MenuItem("Small", callback=lambda _: self.change_model("small"))
        )
        model_menu.add(
            rumps.MenuItem("Medium", callback=lambda _: self.change_model("medium"))
        )
        model_menu.add(
            rumps.MenuItem("Large", callback=lambda _: self.change_model("large"))
        )
        self.menu.add(model_menu)

        # Stats
        self.stats_item = rumps.MenuItem("Statistics...", callback=self.show_stats)
        self.menu.add(self.stats_item)
        self.menu.add(rumps.separator)

        # About
        self.menu.add(rumps.MenuItem("About Kuiskaus", callback=self.show_about))
        self.menu.add(rumps.separator)

        # Quit
        self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def start_hotkey_listener(self):
        """Start the hotkey listener in a background thread"""
        print("[DEBUG] Starting hotkey listener...")

        def run_listener():
            if self.hotkey_listener.start():
                print("[DEBUG] Hotkey listener started successfully")
            else:
                # Can't show alert from background thread, just log
                print("\n⚠️  Accessibility Permission Required!")
                print("Please grant accessibility permissions:")
                print("1. Open System Settings > Privacy & Security > Accessibility")
                print("2. Add Terminal (or your terminal app) to the list")
                print("3. Make sure it's enabled")
                print("4. Restart Kuiskaus")

        # Try starting on main thread instead of background thread
        # since event monitors might need to be on main thread
        run_listener()

    @rumps.clicked("Enabled")
    def toggle_enabled(self, sender):
        """Toggle enabled state"""
        self.enabled = not self.enabled
        sender.state = self.enabled

        if self.enabled:
            self.title = "🎤"
            # Re-enabling is a deliberate user action but it does not fix
            # the microphone: a persisted error (#16) must survive this
            # toggle too, else disable->enable is a silent-clear backdoor.
            if self.audio_recorder.last_error:
                self.update_status(f"🔴 {self.audio_recorder.last_error}")
            else:
                self.update_status("🟢 Ready")
            print("✅ Kuiskaus enabled")
        else:
            self.title = "🔇"
            self.update_status("🔴 Disabled")
            print("🔴 Kuiskaus disabled")

    def toggle_apfel(self, sender: "rumps.MenuItem") -> None:
        """Toggle apfel LLM cleanup"""
        with self._apfel_lock:
            self.use_apfel = not self.use_apfel
            enabled = self.use_apfel
        sender.state = enabled
        status = "enabled" if enabled else "disabled"
        print(f"✨ LLM cleanup {status}")

    def on_hotkey_press(self):
        """Called when hotkey is pressed"""
        if not self.enabled:
            return

        if not self.is_recording:
            admitted = self.audio_recorder.start_recording()
            if not admitted:
                # A worker from a previous recording is still alive (#16).
                # Refuse rather than desyncing app-level state from the
                # recorder, which has no active worker to serve this press.
                return
            self.is_recording = True
            self.recording_start_time = time.time()

            # Update UI. Press only acknowledges the request: the mic
            # is not necessarily capturing yet (PyAudio init), so the
            # live-recording state waits for the capture-started event
            # (issue #43).
            self.title = "🟠"
            self.update_status("🟠 Starting...")

    def _enqueue_capture_started(self) -> None:
        """Capture-started callback from the recorder's worker thread.

        Queues the event's generation only; the rumps.Timer trampoline
        (_drain_ui_events) performs the actual UI transition on the main
        thread (issue #43). No UI mutation here.
        """
        self._pending_capture_started_events.put(self.audio_recorder.current_generation)

    def _drain_ui_events(self, _sender) -> None:
        """Main-thread trampoline: transition to live recording once the
        capture-started event for the current recording generation has
        arrived. Stale events (a newer generation superseded, or the
        recording already released) are dropped so a late worker event
        can never flicker 🔴 over Processing/Ready (issue #43)."""
        while True:
            try:
                event_gen = self._pending_capture_started_events.get_nowait()
            except queue.Empty:
                return
            if (
                not self.audio_recorder.recording
                or self.audio_recorder.current_generation != event_gen
            ):
                continue
            self.title = "🔴"
            self.update_status("🔴 Recording")

    def on_hotkey_release(self):
        """Called when hotkey is released"""
        if not self.enabled:
            return

        if not self.is_recording:
            # A release that finds no active recording is a harmless no-op
            # (issue #17). This branch is reachable even for a physically
            # paired press: when start_recording() refuses admission
            # (#16, prior worker still alive), on_hotkey_press returns
            # early without ever setting is_recording True, so the release
            # for that same keypress lands here. A persisted mic error
            # must therefore survive this branch too -- it clears only on
            # a genuinely successful recording, never on a no-op release.
            self.recording_start_time = None
            self.title = "🎤"
            if self.audio_recorder.last_error:
                self.update_status(f"🔴 {self.audio_recorder.last_error}")
            else:
                self.update_status("🟢 Ready")
            return

        self.is_recording = False
        start_time = self.recording_start_time
        self.recording_start_time = None

        # Stop recording and get audio
        audio_data = self.audio_recorder.stop_recording()

        # A failed/stuck microphone open must surface as its own error and
        # persist until the next successful recording -- no auto-clear
        # timer, and it must NOT be masked as a silent return to Ready (#16).
        if self.audio_recorder.last_error:
            error_msg = self.audio_recorder.last_error
            print(f"⚠️  {error_msg}")
            self.title = "🎤"
            self.update_status(f"🔴 {error_msg}")
            return

        if len(audio_data) == 0:
            # Race-lost: the recording ended before any audio bytes were
            # captured and no mic error was set (#40). Surface it as its
            # own state -- distinct from Ready -- instead of flipping
            # straight back to Ready with no explanation. There is no
            # auto-clear: any subsequent release that yields audio
            # (below) restores the normal flow, and a no-op release
            # (is_recording False) intentionally resets to Ready too,
            # since any later release clears the transient state.
            print("⚠️  No audio captured")
            self.title = "🎤"
            self.update_status("⚪ No audio captured")
            return

        if start_time is None:
            print("⚠️  No start time recorded — ignoring release")
            self.title = "🎤"
            self.update_status("🟢 Ready")
            return
        recording_duration = time.time() - start_time

        # Update UI. A previous race-lost release may have left the ⚪
        # status live; a release that yields audio supersedes it.
        self.title = "🎤"
        self.update_status("🟡 Processing...")

        # Transcribe in a separate thread
        threading.Thread(
            target=self._transcribe_and_insert,
            args=(audio_data, recording_duration),
        ).start()

    def _transcribe_and_insert(
        self, audio_data: np.ndarray, recording_duration: float
    ) -> None:
        """Transcribe audio and insert text (runs in separate thread)"""
        try:
            # Hold the lock across transcribe() so _reload_model cannot
            # cleanup() the model mid-inference (issue #22).
            with self._transcriber_lock:
                transcriber = self.transcriber
                result = transcriber.transcribe(audio_data)
            text = result.get("text", "").strip()

            # Apply apfel cleanup if enabled
            with self._apfel_lock:
                should_clean = self.use_apfel
            if text and should_clean:
                text = clean_with_apfel(text)

            if text:
                # Update stats
                self.total_transcriptions += 1
                self.total_recording_time += recording_duration

                # Insert text
                self.text_inserter.insert_text(text)

                # Log instead of notification (avoids Info.plist issues)
                print(f"📝 Transcribed: {text}")

                # This thread was spawned for a recording that completed
                # without a mic error. But transcription runs async and can
                # outlive a *later* press/release cycle; if that later
                # cycle has since set last_error, this stale completion
                # must not clobber it (#16, same principle as the no-op
                # release branch above).
                if not self.audio_recorder.last_error:
                    self.update_status("🟢 Ready")
            elif not self.audio_recorder.last_error:
                self.update_status("🟢 Ready (no speech)")

        # Top-level guard for the transcription worker thread: any failure
        # here (model, inference, text insertion) must not crash the app.
        except Exception as e:  # noqa: BLE001 - logged; worker-thread guard
            print(f"Error during transcription: {e}")
            if not self.audio_recorder.last_error:
                self.update_status("🟢 Ready (error)")

    def update_status(self, status: str):
        """Update the status menu item"""
        self.status_item.title = status

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

    @rumps.clicked("Statistics...")
    def show_stats(self, _):
        """Show statistics dialog"""
        session_duration = (_utcnow() - self.session_start).total_seconds()
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)

        stats_text = f"""Session Duration: {hours}h {minutes}m
Total Transcriptions: {self.total_transcriptions}
Total Recording Time: {self.total_recording_time:.1f}s
Average Recording: {self.total_recording_time / max(1, self.total_transcriptions):.1f}s"""

        rumps.alert("Kuiskaus Statistics", stats_text)

    @rumps.clicked("About Kuiskaus")
    def show_about(self, _):
        """Show about dialog"""
        about_text = """Kuiskaus - Speech-to-Text for macOS

A lightweight menu bar app that uses OpenAI's Whisper V3 Turbo model for fast, accurate speech-to-text conversion.

Hold Control+Option (⌃⌥) to record, release to transcribe and insert text at your cursor position.

Version 1.0
© 2024"""

        rumps.alert("About Kuiskaus", about_text)

    @rumps.clicked("Quit")
    def quit_app(self, _):
        """Quit the application"""
        # Show stats before quitting
        if self.total_transcriptions > 0:
            response = rumps.alert(
                "Quit Kuiskaus?",
                f"You've made {self.total_transcriptions} transcriptions this session.",
                ok="Quit",
                cancel="Cancel",
            )
            if response == 0:  # Cancel
                return

        # Cleanup
        try:
            self._ui_tick_timer.stop()
        except Exception as e:  # noqa: BLE001 - logged; must not raise from quit
            print(f"Timer stop error: {e}")
        self.cleanup()
        rumps.quit_application()

    def cleanup(self):
        """Clean up resources"""
        try:
            self.hotkey_listener.stop()
            self.audio_recorder.cleanup()
            with self._transcriber_lock:
                transcriber = self.transcriber
            # The transcriber cleanup runs OUTSIDE the transcriber lock:
            # for background-load transcribers (Parakeet, Voxtral)
            # cleanup() waits on the load thread, and that load can be a
            # multi-minute model download — holding the lock while
            # waiting would wedge every in-flight transcription worker
            # (and therefore quit) on the same in-flight-load hazard as
            # the reload path (issue #22 review). A worker still running
            # keeps its own snapshot, so releasing the model out of lock
            # is safe.
            transcriber.cleanup()
        except Exception as e:  # noqa: BLE001 - logged; must not raise from quit
            print(f"Cleanup error: {e}")


def main():
    """Main entry point for menu bar app"""
    # Check for Apple Silicon
    if not check_apple_silicon():
        rumps.alert(
            "Apple Silicon Required",
            "This application requires Apple Silicon (M1/M2/M3). Intel-based Macs are not supported.",
            ok="OK",
        )
        return

    app = KuiskausMenuBarApp()
    app.run()


if __name__ == "__main__":
    main()
