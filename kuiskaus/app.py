#!/usr/bin/env python3
"""
Kuiskaus - Whisper V3 Turbo Speech-to-Text for macOS
Hold Control+Option to record, release to transcribe and insert text.
"""

import sys
import threading
import time

import numpy as np

from .audio_recorder import AudioRecorder
from .hotkey_listener import HotkeyListener
from .parakeet_transcriber import ParakeetTranscriber
from .postprocessor import clean_with_apfel
from .silicon_check import check_apple_silicon
from .text_inserter import TextInserter
from .transcriber import Transcriber
from .whisper_transcriber import WhisperTranscriber

# Optional: notifications
try:
    from Foundation import NSUserNotification, NSUserNotificationCenter

    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False


ALLOWED_MODELS = {"turbo", "base", "small", "medium", "large", "parakeet", "voxtral"}


class KuiskausApp:
    def __init__(self, model_name: str = "parakeet", use_apfel: bool = False):
        """
        Initialize the Kuiskaus application

        Args:
            model_name: STT model to use (default: "parakeet")
            use_apfel: Whether to use LLM cleanup with apfel (default: False)
        """
        print("Initializing Kuiskaus...")

        # Initialize components
        self.audio_recorder = AudioRecorder()
        if model_name == "parakeet":
            self.transcriber: Transcriber = ParakeetTranscriber()
        elif model_name == "voxtral":
            from .voxtral_transcriber import VoxtralTranscriber

            self.transcriber: Transcriber = VoxtralTranscriber()
        else:
            self.transcriber: Transcriber = WhisperTranscriber(model_name=model_name)
        if not isinstance(self.transcriber, Transcriber):
            raise TypeError(
                f"Transcriber implementation {type(self.transcriber)} does not satisfy "
                "the Transcriber protocol"
            )
        self.text_inserter = TextInserter()

        # State
        self.is_recording = False
        self.recording_start_time = None
        self.use_apfel = use_apfel

        # Initialize hotkey listener with callbacks
        self.hotkey_listener = HotkeyListener(
            on_press=self.on_hotkey_press, on_release=self.on_hotkey_release
        )

        # Stats
        self.total_transcriptions = 0
        self.total_recording_time = 0.0

    def on_hotkey_press(self):
        """Called when hotkey is pressed"""
        if not self.is_recording:
            admitted = self.audio_recorder.start_recording()
            if not admitted:
                # A worker from a previous recording is still alive (#16).
                # Refuse rather than desyncing app-level state from the
                # recorder, which has no active worker to serve this press.
                print(
                    "⚠️  Recording could not start — previous recording still stopping"
                )
                return
            self.is_recording = True
            self.recording_start_time = time.time()
            print("🎤 Recording...")
            self.show_notification("Recording", "Speak now...")

    def on_hotkey_release(self):
        """Called when hotkey is released"""
        if not self.is_recording:
            # No active recording: release is a harmless no-op (#17).
            # Never call stop_recording() without a matching press.
            self.recording_start_time = None
            print("✅ Ready")
            return
        self.is_recording = False
        start_time = self.recording_start_time
        self.recording_start_time = None

        # Stop recording and get audio
        audio_data = self.audio_recorder.stop_recording()

        # A failed/stuck microphone open must surface as its own error,
        # not as "no audio recorded" -- that's indistinguishable from
        # genuine silence (#16).
        if self.audio_recorder.last_error:
            error_msg = self.audio_recorder.last_error
            print(f"⚠️  {error_msg}")
            self.show_notification("Error", error_msg)
            return

        if start_time is None:
            print("⚠️  No start time recorded — ignoring release")
            self.show_notification(
                "Recording error", "Missing start time; recording discarded"
            )
            return
        recording_duration = time.time() - start_time

        print(f"⏹️  Stopped recording ({recording_duration:.1f}s)")

        if len(audio_data) > 0:
            # Transcribe in a separate thread to avoid blocking
            threading.Thread(
                target=self._transcribe_and_insert,
                args=(audio_data, recording_duration),
            ).start()
        else:
            print("No audio recorded")

    def _transcribe_and_insert(
        self, audio_data: np.ndarray, recording_duration: float
    ) -> None:
        """Transcribe audio and insert text (runs in separate thread)"""
        try:
            print("🤖 Transcribing...")
            self.show_notification("Transcribing", "Processing your speech...")

            # Transcribe
            result = self.transcriber.transcribe(audio_data)
            text = result.get("text", "").strip()

            if text:
                # Apply apfel cleanup if enabled
                if self.use_apfel:
                    text = clean_with_apfel(text)

                # Update stats
                self.total_transcriptions += 1
                self.total_recording_time += recording_duration

                # Log result
                print(f"📝 Transcribed: {text}")
                if "transcribe_time" in result:
                    rtf = result.get("rtf", 0)
                    print(
                        f"   Performance: {result['transcribe_time']:.2f}s ({rtf:.2f}x realtime)"
                    )

                # Insert text
                self.text_inserter.insert_text(text)
                self.show_notification(
                    "Transcribed", text[:50] + "..." if len(text) > 50 else text
                )
            else:
                print("No speech detected")
                self.show_notification(
                    "No speech detected", "Try speaking more clearly"
                )

        # Top-level guard for the transcription worker thread: any failure
        # here (model, inference, text insertion) must not crash the app.
        except Exception as e:  # noqa: BLE001 - logged; worker-thread guard
            print(f"Error during transcription: {e}")
            self.show_notification("Error", "Failed to transcribe audio")

    def show_notification(self, title: str, message: str):
        """Show a macOS notification"""
        if HAS_NOTIFICATIONS:
            try:
                notification = NSUserNotification.alloc().init()
                notification.setTitle_(title)
                notification.setInformativeText_(message)
                NSUserNotificationCenter.defaultUserNotificationCenter().deliverNotification_(
                    notification
                )
            except (AttributeError, OSError, RuntimeError) as e:
                # Notification failure must never break the transcription flow.
                # AttributeError covers defaultUserNotificationCenter() being
                # None when no AppKit run loop is active (CLI path).
                print(f"Failed to show notification: {e}")

    def print_stats(self):
        """Print usage statistics"""
        print("\n📊 Session Statistics:")
        print(f"   Total transcriptions: {self.total_transcriptions}")
        print(f"   Total recording time: {self.total_recording_time:.1f}s")
        if self.total_transcriptions > 0:
            avg_duration = self.total_recording_time / self.total_transcriptions
            print(f"   Average recording: {avg_duration:.1f}s")

    def run(self):
        """Run the application"""
        print("\n🚀 Kuiskaus is running!")
        print("📌 Hold Control+Option (⌃⌥) to record")
        print("📌 Release to transcribe and insert text")
        print("📌 Press Ctrl+C to quit\n")

        # Start hotkey listener
        if not self.hotkey_listener.start():
            print("\n❌ Failed to start hotkey listener")
            print("Please grant accessibility permissions and restart")
            return

        try:
            # Run event loop (blocks)
            self.hotkey_listener.run_loop()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.hotkey_listener.stop()
        self.audio_recorder.cleanup()
        self.transcriber.cleanup()
        self.print_stats()
        print("👋 Goodbye!")


def main():
    """Main entry point"""
    # Check for Apple Silicon
    if not check_apple_silicon():
        print("\n❌ Error: This application requires Apple Silicon (M1/M2/M3)")
        print("Intel-based Macs are not supported.")
        sys.exit(1)

    # Parse arguments (simple for now)
    model_name = "parakeet"
    use_apfel = "--apfel" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--apfel"]
    if args:
        model_name = args[0]
        if model_name not in ALLOWED_MODELS:
            print(
                f"❌ Unknown model '{model_name}'. Allowed: {', '.join(sorted(ALLOWED_MODELS))}"
            )
            sys.exit(1)
    print(f"Using model: {model_name}")
    if use_apfel:
        print("LLM cleanup enabled (apfel)")

    # Create and run app
    app = KuiskausApp(model_name=model_name, use_apfel=use_apfel)
    app.run()


if __name__ == "__main__":
    main()
