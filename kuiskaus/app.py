#!/usr/bin/env python3
"""
Kuiskaus - Whisper V3 Turbo Speech-to-Text for macOS
Hold Control+Option to record, release to transcribe and insert text.
"""

import sys
import os
import threading
import time
from datetime import datetime

from .audio_recorder import AudioRecorder
from .whisper_transcriber import WhisperTranscriber
from .hotkey_listener import HotkeyListener
from .text_inserter import TextInserter

# Optional: notifications
try:
    from Foundation import NSUserNotification, NSUserNotificationCenter
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False

class KuiskausApp:
    def __init__(self, model_name: str = "turbo"):
        """
        Initialize the Kuiskaus application
        
        Args:
            model_name: Whisper model to use (default: "turbo" for V3 Turbo)
        """
        print("Initializing Kuiskaus...")
        
        # Initialize components
        self.audio_recorder = AudioRecorder()
        self.transcriber = WhisperTranscriber(model_name=model_name)
        self.text_inserter = TextInserter()
        
        # State
        self.is_recording = False
        self.recording_start_time = None
        
        # Initialize hotkey listener with callbacks
        self.hotkey_listener = HotkeyListener(
            on_press=self.on_hotkey_press,
            on_release=self.on_hotkey_release
        )
        
        # Stats
        self.total_transcriptions = 0
        self.total_recording_time = 0.0
        
    def on_hotkey_press(self):
        """Called when hotkey is pressed"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.audio_recorder.start_recording()
            print("🎤 Recording...")
            self.show_notification("Recording", "Speak now...")
    
    def on_hotkey_release(self):
        """Called when hotkey is released"""
        if self.is_recording:
            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time
            
            print(f"⏹️  Stopped recording ({recording_duration:.1f}s)")
            
            # Stop recording and get audio
            audio_data = self.audio_recorder.stop_recording()
            
            if len(audio_data) > 0:
                # Transcribe in a separate thread to avoid blocking
                threading.Thread(
                    target=self._transcribe_and_insert,
                    args=(audio_data, recording_duration)
                ).start()
            else:
                print("No audio recorded")
    
    def _transcribe_and_insert(self, audio_data, recording_duration):
        """Transcribe audio and insert text (runs in separate thread)"""
        try:
            print("🤖 Transcribing...")
            self.show_notification("Transcribing", "Processing your speech...")
            
            # Transcribe
            result = self.transcriber.transcribe(audio_data)
            text = result.get("text", "").strip()
            
            if text:
                # Update stats
                self.total_transcriptions += 1
                self.total_recording_time += recording_duration
                
                # Log result
                print(f"📝 Transcribed: {text}")
                if "transcribe_time" in result:
                    rtf = result.get("rtf", 0)
                    print(f"   Performance: {result['transcribe_time']:.2f}s ({rtf:.2f}x realtime)")
                
                # Insert text
                self.text_inserter.insert_text(text)
                self.show_notification("Transcribed", text[:50] + "..." if len(text) > 50 else text)
            else:
                print("No speech detected")
                self.show_notification("No speech detected", "Try speaking more clearly")
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            self.show_notification("Error", "Failed to transcribe audio")
    
    def show_notification(self, title: str, message: str):
        """Show a macOS notification"""
        if HAS_NOTIFICATIONS:
            try:
                notification = NSUserNotification.alloc().init()
                notification.setTitle_(title)
                notification.setInformativeText_(message)
                NSUserNotificationCenter.defaultUserNotificationCenter().deliverNotification_(notification)
            except Exception as e:
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

def check_apple_silicon():
    """Check if running on Apple Silicon"""
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                              capture_output=True, text=True)
        return "Apple" in result.stdout
    except:
        return False

def main():
    """Main entry point"""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check for Apple Silicon
    if not check_apple_silicon():
        print("\n❌ Error: This application requires Apple Silicon (M1/M2/M3)")
        print("Intel-based Macs are not supported.")
        sys.exit(1)
    
    # Parse arguments (simple for now)
    model_name = "turbo"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"Using model: {model_name}")
    
    # Create and run app
    app = KuiskausApp(model_name=model_name)
    app.run()

if __name__ == "__main__":
    main()