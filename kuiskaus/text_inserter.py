import Quartz
import time
from AppKit import NSPasteboard, NSPasteboardTypeString
import threading

class TextInserter:
    def __init__(self):
        """Initialize text inserter"""
        self.insert_lock = threading.Lock()
        
    def insert_text_typing(self, text: str, delay: float = 0.001):
        """
        Insert text by simulating keyboard typing
        
        Args:
            text: Text to insert
            delay: Delay between keystrokes (seconds)
        """
        with self.insert_lock:
            # Small delay to ensure we're ready
            time.sleep(0.1)
            
            for char in text:
                self._type_character(char)
                if delay > 0:
                    time.sleep(delay)
    
    def insert_text_paste(self, text: str):
        """
        Insert text using clipboard paste (faster for long text)
        
        Args:
            text: Text to insert
        """
        with self.insert_lock:
            # Save current clipboard content
            pasteboard = NSPasteboard.generalPasteboard()
            old_content = pasteboard.stringForType_(NSPasteboardTypeString)
            
            try:
                # Set new clipboard content
                pasteboard.clearContents()
                pasteboard.setString_forType_(text, NSPasteboardTypeString)
                
                # Small delay to ensure clipboard is updated
                time.sleep(0.05)
                
                # Simulate Cmd+V
                self._simulate_paste()
                
                # Small delay to ensure paste completes
                time.sleep(0.1)
                
            finally:
                # Restore original clipboard content
                if old_content is not None:
                    pasteboard.clearContents()
                    pasteboard.setString_forType_(old_content, NSPasteboardTypeString)
    
    def _type_character(self, char: str):
        """Type a single character"""
        # Get the Unicode value
        unicode_value = ord(char)
        
        # Create key down event
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)
        
        # Create key up event
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)
    
    def _simulate_paste(self):
        """Simulate Cmd+V key combination"""
        # Key code for 'v' is 9
        v_key = 9
        
        # Press Cmd key
        cmd_down = Quartz.CGEventCreateKeyboardEvent(None, 0x37, True)  # 0x37 is Command key
        Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_down)
        
        # Press 'v' with Cmd held
        v_down = Quartz.CGEventCreateKeyboardEvent(None, v_key, True)
        Quartz.CGEventSetFlags(v_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_down)
        
        # Release 'v'
        v_up = Quartz.CGEventCreateKeyboardEvent(None, v_key, False)
        Quartz.CGEventSetFlags(v_up, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_up)
        
        # Release Cmd
        cmd_up = Quartz.CGEventCreateKeyboardEvent(None, 0x37, False)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_up)
    
    def insert_text(self, text: str, use_paste: bool = True):
        """
        Insert text at current cursor position
        
        Args:
            text: Text to insert
            use_paste: If True, use clipboard paste (faster). If False, type character by character.
        """
        if not text:
            return
            
        if use_paste and len(text) > 10:  # Use paste for longer text
            self.insert_text_paste(text)
        else:
            self.insert_text_typing(text)