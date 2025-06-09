import Quartz
from AppKit import NSEvent
import threading
from typing import Callable, Optional

class HotkeyListenerCGEvent:
    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None]):
        """
        Initialize hotkey listener using CGEventTap
        
        Args:
            on_press: Function to call when hotkey is pressed
            on_release: Function to call when hotkey is released
        """
        self.on_press = on_press
        self.on_release = on_release
        
        # Track state
        self.is_pressed = False
        self.tap = None
        self.run_loop_source = None
        self.running = False
        
    def _check_modifiers(self, flags: int) -> bool:
        """Check if the required modifier keys are pressed"""
        # CGEventFlags values
        kCGEventFlagMaskControl = 1 << 18
        kCGEventFlagMaskAlternate = 1 << 19  # Option key
        kCGEventFlagMaskCommand = 1 << 20
        
        # Check Control key
        has_control = bool(flags & kCGEventFlagMaskControl)
        # Check Option key
        has_option = bool(flags & kCGEventFlagMaskAlternate)
        # Check that Command is NOT pressed (to avoid conflicts)
        has_command = bool(flags & kCGEventFlagMaskCommand)
        
        return has_control and has_option and not has_command
    
    def _event_tap_callback(self, proxy, type_, event, refcon):
        """CGEventTap callback"""
        try:
            # Check if it's a flags changed event
            if type_ == Quartz.kCGEventFlagsChanged:
                flags = Quartz.CGEventGetFlags(event)
                modifiers_pressed = self._check_modifiers(flags)
                
                # Debug output
                if flags != 0:
                    print(f"[DEBUG CGEvent] Modifier flags: {flags}, Control+Option pressed: {modifiers_pressed}")
                
                if modifiers_pressed and not self.is_pressed:
                    # Hotkey pressed
                    print("[DEBUG CGEvent] Hotkey pressed!")
                    self.is_pressed = True
                    if self.on_press:
                        # Run callback in separate thread to avoid blocking
                        threading.Thread(target=self.on_press).start()
                        
                elif not modifiers_pressed and self.is_pressed:
                    # Hotkey released
                    print("[DEBUG CGEvent] Hotkey released!")
                    self.is_pressed = False
                    if self.on_release:
                        # Run callback in separate thread to avoid blocking
                        threading.Thread(target=self.on_release).start()
            
        except Exception as e:
            print(f"Error in event tap callback: {e}")
        
        # Return the event to continue processing
        return event
    
    def start(self):
        """Start listening for hotkeys"""
        if not self.running:
            self.running = True
            
            # Check accessibility permissions
            from ApplicationServices import AXIsProcessTrusted
            is_trusted = AXIsProcessTrusted()
            
            if not is_trusted:
                print("Accessibility permissions required!")
                print("Please grant accessibility permissions in System Preferences > Security & Privacy > Accessibility")
                return False
            
            print("[DEBUG CGEvent] Creating CGEventTap...")
            
            # Create event tap
            self.tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,  # Session level
                Quartz.kCGHeadInsertEventTap,  # Insert at head
                Quartz.kCGEventTapOptionListenOnly,  # Just listen, don't modify
                Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),  # Monitor modifier key changes
                self._event_tap_callback,
                None  # refcon
            )
            
            if not self.tap:
                print("[DEBUG CGEvent] Failed to create CGEventTap!")
                return False
            
            print("[DEBUG CGEvent] CGEventTap created successfully")
            
            # Create run loop source
            self.run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
            
            # Add to current run loop
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetCurrent(),
                self.run_loop_source,
                Quartz.kCFRunLoopDefaultMode
            )
            
            # Enable the tap
            Quartz.CGEventTapEnable(self.tap, True)
            
            print("[DEBUG CGEvent] Hotkey listener started. Press Control+Option (⌃⌥) to record.")
            return True
    
    def stop(self):
        """Stop listening for hotkeys"""
        if self.running:
            if self.tap:
                Quartz.CGEventTapEnable(self.tap, False)
                if self.run_loop_source:
                    Quartz.CFRunLoopRemoveSource(
                        Quartz.CFRunLoopGetCurrent(),
                        self.run_loop_source,
                        Quartz.kCFRunLoopDefaultMode
                    )
                self.tap = None
                self.run_loop_source = None
            
            self.running = False
            self.is_pressed = False
            print("[DEBUG CGEvent] Hotkey listener stopped.")
    
    def run_loop(self):
        """Run the event loop (blocks) - for CLI app"""
        try:
            Quartz.CFRunLoopRun()
        except KeyboardInterrupt:
            self.stop()
    
    def stop_loop(self):
        """Stop the event loop"""
        Quartz.CFRunLoopStop(Quartz.CFRunLoopGetCurrent())