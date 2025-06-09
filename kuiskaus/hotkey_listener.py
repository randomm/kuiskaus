import Quartz
from AppKit import NSEvent, NSApplication, NSApp
from PyObjCTools import AppHelper
import threading
from typing import Callable, Optional, Set

# Define event masks
NSKeyDownMask = 1 << 10
NSKeyUpMask = 1 << 11
NSFlagsChangedMask = 1 << 12

class HotkeyListener:
    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None]):
        """
        Initialize hotkey listener
        
        Args:
            on_press: Function to call when hotkey is pressed
            on_release: Function to call when hotkey is released
        """
        self.on_press = on_press
        self.on_release = on_release
        
        # Default to Control+Option (⌃⌥) as modifier
        # We'll check modifiers by their bit masks directly
        self.required_modifiers = None  # Not used anymore
        
        # Track state
        self.is_pressed = False
        self.monitor = None
        self.running = False
        
    def _check_modifiers(self, flags: int) -> bool:
        """Check if the required modifier keys are pressed"""
        # ModifierFlags values
        NSControlKeyMask = 1 << 18
        NSAlternateKeyMask = 1 << 19  # Option key
        NSCommandKeyMask = 1 << 20
        
        # Check Control key
        has_control = bool(flags & NSControlKeyMask)
        # Check Option key
        has_option = bool(flags & NSAlternateKeyMask)
        # Check that Command is NOT pressed (to avoid conflicts)
        has_command = bool(flags & NSCommandKeyMask)
        
        return has_control and has_option and not has_command
    
    def _handle_event(self, event):
        """Handle keyboard events"""
        print(f"[DEBUG] Event handler called! Type: {event.type() if event else 'None'}")
        try:
            event_type = event.type()
            flags = event.modifierFlags()
            
            print(f"[DEBUG] Event type: {event_type}, NSFlagsChangedMask: {NSFlagsChangedMask}")
            
            if event_type == NSFlagsChangedMask:
                # Modifier key changed
                modifiers_pressed = self._check_modifiers(flags)
                
                # Debug: print which modifiers are pressed
                print(f"[DEBUG] Modifier flags: {flags}, Control+Option pressed: {modifiers_pressed}")
                
                if modifiers_pressed and not self.is_pressed:
                    # Hotkey pressed
                    print("[DEBUG] Hotkey pressed!")
                    self.is_pressed = True
                    if self.on_press:
                        # Run callback in separate thread to avoid blocking
                        threading.Thread(target=self.on_press).start()
                        
                elif not modifiers_pressed and self.is_pressed:
                    # Hotkey released
                    print("[DEBUG] Hotkey released!")
                    self.is_pressed = False
                    if self.on_release:
                        # Run callback in separate thread to avoid blocking
                        threading.Thread(target=self.on_release).start()
            
        except Exception as e:
            print(f"Error handling event: {e}")
        
        # Return the event to continue processing
        return event
    
    def start(self):
        """Start listening for hotkeys"""
        if not self.running:
            self.running = True
            
            # Check accessibility permissions
            # Use the function from ApplicationServices
            from ApplicationServices import AXIsProcessTrusted
            is_trusted = AXIsProcessTrusted()
            
            if not is_trusted:
                print("Accessibility permissions required!")
                print("Please grant accessibility permissions in System Preferences > Security & Privacy > Accessibility")
                return False
            
            # Create event monitor
            mask = NSFlagsChangedMask
            print(f"[DEBUG] Creating global event monitor with mask: {mask}")
            
            # For rumps/menu bar apps, we need to run on main thread
            # Try using local monitor as well as global
            self.local_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                mask, self._handle_event
            )
            
            self.monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask, self._handle_event
            )
            
            if self.monitor or self.local_monitor:
                print(f"[DEBUG] Event monitors created - Global: {self.monitor is not None}, Local: {self.local_monitor is not None}")
                print("Hotkey listener started. Press Control+Option (⌃⌥) to record.")
                return True
            else:
                print("[DEBUG] Failed to create event monitors!")
                return False
    
    def stop(self):
        """Stop listening for hotkeys"""
        if self.running:
            if self.monitor:
                NSEvent.removeMonitor_(self.monitor)
                self.monitor = None
            if hasattr(self, 'local_monitor') and self.local_monitor:
                NSEvent.removeMonitor_(self.local_monitor)
                self.local_monitor = None
            self.running = False
            self.is_pressed = False
            print("Hotkey listener stopped.")
    
    def run_loop(self):
        """Run the event loop (blocks)"""
        try:
            AppHelper.runEventLoop()
        except KeyboardInterrupt:
            self.stop()
    
    def stop_loop(self):
        """Stop the event loop"""
        AppHelper.stopEventLoop()