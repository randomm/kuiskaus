#!/usr/bin/env python3
"""
Kuiskaus Menu Bar App - Whisper V3 Turbo Speech-to-Text for macOS
A menu bar application for easy access to speech-to-text functionality.
"""

import rumps
import threading
import time
from datetime import datetime

from .audio_recorder import AudioRecorder
from .whisper_transcriber import WhisperTranscriber
from .hotkey_listener_cgevent import HotkeyListenerCGEvent
from .text_inserter import TextInserter

class KuiskausMenuBarApp(rumps.App):
    def __init__(self):
        super(KuiskausMenuBarApp, self).__init__(
            "Kuiskaus",
            title="🎤",  # Use title instead of icon for emoji
            quit_button=None  # We'll add custom quit with stats
        )
        
        # Initialize components
        self.audio_recorder = AudioRecorder()
        self.transcriber = WhisperTranscriber(model_name="turbo")
        self.text_inserter = TextInserter()
        
        # State
        self.is_recording = False
        self.recording_start_time = None
        self.enabled = True
        
        # Initialize hotkey listener with CGEventTap
        self.hotkey_listener = HotkeyListenerCGEvent(
            on_press=self.on_hotkey_press,
            on_release=self.on_hotkey_release
        )
        
        # Stats
        self.total_transcriptions = 0
        self.total_recording_time = 0.0
        self.session_start = datetime.now()
        
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
        
        # Hotkey info
        self.menu.add(rumps.MenuItem("Hotkey: ⌃⌥ (Control+Option)", callback=None))
        self.menu.add(rumps.separator)
        
        # Model selection submenu
        model_menu = rumps.MenuItem("Model")
        model_menu.add(rumps.MenuItem("Turbo (Fastest)", callback=lambda _: self.change_model("turbo")))
        model_menu.add(rumps.MenuItem("Base", callback=lambda _: self.change_model("base")))
        model_menu.add(rumps.MenuItem("Small", callback=lambda _: self.change_model("small")))
        model_menu.add(rumps.MenuItem("Medium", callback=lambda _: self.change_model("medium")))
        model_menu.add(rumps.MenuItem("Large", callback=lambda _: self.change_model("large")))
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
            self.update_status("🟢 Ready")
            print("✅ Kuiskaus enabled")
        else:
            self.title = "🔇"
            self.update_status("🔴 Disabled")
            print("🔴 Kuiskaus disabled")
    
    def on_hotkey_press(self):
        """Called when hotkey is pressed"""
        if not self.enabled:
            return
            
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.audio_recorder.start_recording()
            
            # Update UI
            self.title = "🔴"
            self.update_status("🔴 Recording...")
    
    def on_hotkey_release(self):
        """Called when hotkey is released"""
        if not self.enabled:
            return
            
        if self.is_recording:
            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time
            
            # Update UI
            self.title = "🎤"
            self.update_status("🟡 Processing...")
            
            # Stop recording and get audio
            audio_data = self.audio_recorder.stop_recording()
            
            if len(audio_data) > 0:
                # Transcribe in a separate thread
                threading.Thread(
                    target=self._transcribe_and_insert,
                    args=(audio_data, recording_duration)
                ).start()
            else:
                self.update_status("🟢 Ready")
    
    def _transcribe_and_insert(self, audio_data, recording_duration):
        """Transcribe audio and insert text (runs in separate thread)"""
        try:
            # Transcribe
            result = self.transcriber.transcribe(audio_data)
            text = result.get("text", "").strip()
            
            if text:
                # Update stats
                self.total_transcriptions += 1
                self.total_recording_time += recording_duration
                
                # Insert text
                self.text_inserter.insert_text(text)
                
                # Log instead of notification (avoids Info.plist issues)
                print(f"📝 Transcribed: {text}")
                
                self.update_status("🟢 Ready")
            else:
                self.update_status("🟢 Ready (no speech)")
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            self.update_status("🟢 Ready (error)")
    
    def update_status(self, status: str):
        """Update the status menu item"""
        self.status_item.title = status
    
    def change_model(self, model_name: str):
        """Change the Whisper model"""
        self.update_status(f"Loading {model_name} model...")
        
        # Reload transcriber with new model
        threading.Thread(
            target=self._reload_model,
            args=(model_name,)
        ).start()
    
    def _reload_model(self, model_name: str):
        """Reload the model in background"""
        try:
            old_transcriber = self.transcriber
            self.transcriber = WhisperTranscriber(model_name=model_name)
            old_transcriber.cleanup()
            
            self.update_status("🟢 Ready")
            print(f"✅ Model changed to {model_name}")
        except Exception as e:
            self.update_status("🟢 Ready (model error)")
            print(f"❌ Model error: {e}")
    
    @rumps.clicked("Statistics...")
    def show_stats(self, _):
        """Show statistics dialog"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
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
                cancel="Cancel"
            )
            if response == 0:  # Cancel
                return
        
        # Cleanup
        self.cleanup()
        rumps.quit_application()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.hotkey_listener.stop()
            self.audio_recorder.cleanup()
            self.transcriber.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")

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
    """Main entry point for menu bar app"""
    # Check for Apple Silicon
    if not check_apple_silicon():
        rumps.alert(
            "Apple Silicon Required",
            "This application requires Apple Silicon (M1/M2/M3). Intel-based Macs are not supported.",
            ok="OK"
        )
        return
    
    app = KuiskausMenuBarApp()
    app.run()

if __name__ == "__main__":
    main()