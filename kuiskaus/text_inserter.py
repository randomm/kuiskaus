import threading
import time

import Quartz
from AppKit import NSPasteboard, NSPasteboardTypeString

# TCC hint messages. AXIsProcessTrusted is a hint, NOT a detector: on
# macOS 26 Tahoe with ad-hoc/uv-Python identities it can return True
# while CGEventPost still fails, so the "trusted" message recommends
# checking Input Injection specifically instead of assuming the grants
# are fine.
_TCC_REVOKED_MESSAGE = (
    "Accessibility permission revoked — re-grant in System Settings > "
    "Privacy & Security > Accessibility (then restart the app)"
)
_TCC_TAHOE_MESSAGE = (
    "Text insertion failed — verify Input Injection AND Accessibility "
    "grants (macOS 26 tracks these separately in System Settings > "
    "Privacy & Security). If both are granted, this may be a Tahoe "
    "silent-drop bug (issue TBD for verification-based detection)."
)


class TextInserter:
    def __init__(self):
        """Initialize text inserter"""
        self.insert_lock = threading.Lock()
        # Error data, never a UI call (audio_recorder.last_error pattern).
        # Cleared at the start of every insert_text() call.
        self.last_error: str | None = None
        # AXIsProcessTrusted is cached after the first failure per
        # session: once trust is revoked it stays revoked until restart,
        # and the check itself can prompt the TCC dialog.
        self._ax_trusted: bool | None = None

    def insert_text_typing(self, text: str, delay: float = 0.001) -> bool:
        """
        Insert text by simulating keyboard typing

        Args:
            text: Text to insert
            delay: Delay between keystrokes (seconds)

        Returns:
            False if any keystroke failed (see last_error), else True.
        """
        with self.insert_lock:
            # Small delay to ensure we're ready
            time.sleep(0.1)

            for char in text:
                if not self._type_character(char):
                    self._surface_tcc_hint()
                    return False
                if delay > 0:
                    time.sleep(delay)
            return True

    def insert_text_paste(self, text: str) -> bool:
        """
        Insert text using clipboard paste (faster for long text)

        Args:
            text: Text to insert

        Returns:
            False if the clipboard write or paste keystrokes failed
            (see last_error), else True.
        """
        with self.insert_lock:
            # Save current clipboard content
            pasteboard = NSPasteboard.generalPasteboard()
            old_content = pasteboard.stringForType_(NSPasteboardTypeString)

            # On failure, the transcribed text stays on the clipboard so
            # the user can recover with a manual Cmd+V — so the prior
            # clipboard is only restored when every step succeeded.
            paste_succeeded = False

            try:
                # Set new clipboard content
                pasteboard.clearContents()
                if not pasteboard.setString_forType_(text, NSPasteboardTypeString):
                    self.last_error = (
                        "Failed to write transcribed text to the "
                        "clipboard — text was not inserted"
                    )
                    return False

                # Small delay to ensure clipboard is updated
                time.sleep(0.05)

                # Simulate Cmd+V
                paste_succeeded = self._simulate_paste()

                # Small delay to ensure paste completes
                time.sleep(0.1)
            finally:
                # Restore original clipboard content (only on success —
                # see the paste_succeeded comment above).
                if paste_succeeded and old_content is not None:
                    pasteboard.clearContents()
                    pasteboard.setString_forType_(old_content, NSPasteboardTypeString)

            if not paste_succeeded:
                self._surface_tcc_hint()
                return False
            return True

    def _type_character(self, char: str) -> bool:
        """Type a single character. Returns False if either event post failed."""
        # Create key down event
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, event):
            self.last_error = "Keyboard event injection failed (key down)"
            return False

        # Create key up event
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, event):
            self.last_error = "Keyboard event injection failed (key up)"
            return False
        return True

    def _simulate_paste(self) -> bool:
        """Simulate Cmd+V key combination. Returns False if any post failed."""
        # Key code for 'v' is 9
        v_key = 9

        # Press Cmd key
        cmd_down = Quartz.CGEventCreateKeyboardEvent(
            None, 0x37, True
        )  # 0x37 is Command key
        Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_down):
            self.last_error = "Keyboard event injection failed (Cmd down)"
            return False

        # Press 'v' with Cmd held
        v_down = Quartz.CGEventCreateKeyboardEvent(None, v_key, True)
        Quartz.CGEventSetFlags(v_down, Quartz.kCGEventFlagMaskCommand)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_down):
            self.last_error = "Keyboard event injection failed (V down)"
            return False

        # Release 'v'
        v_up = Quartz.CGEventCreateKeyboardEvent(None, v_key, False)
        Quartz.CGEventSetFlags(v_up, Quartz.kCGEventFlagMaskCommand)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_up):
            self.last_error = "Keyboard event injection failed (V up)"
            return False

        # Release Cmd
        cmd_up = Quartz.CGEventCreateKeyboardEvent(None, 0x37, False)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_up):
            self.last_error = "Keyboard event injection failed (Cmd up)"
            return False
        return True

    def _surface_tcc_hint(self) -> None:
        """Refine last_error with a TCC hint after an injection failure.

        AXIsProcessTrusted() is cached for the session (trust, once
        revoked, only changes on re-grant + restart). The result is a
        hint for the message text, not a detector — see the module
        comment on the two message constants.
        """
        if self._ax_trusted is None:
            from ApplicationServices import AXIsProcessTrusted

            self._ax_trusted = bool(AXIsProcessTrusted())
        self.last_error = (
            _TCC_REVOKED_MESSAGE if not self._ax_trusted else _TCC_TAHOE_MESSAGE
        )

    def insert_text(self, text: str, use_paste: bool = True) -> bool:
        """
        Insert text at current cursor position

        Args:
            text: Text to insert
            use_paste: If True, use clipboard paste (faster). If False, type character by character.

        Returns:
            False if insertion failed (see last_error), else True.
            Empty text is a success (nothing to insert).
        """
        self.last_error = None
        if not text:
            return True

        if use_paste and len(text) > 10:  # Use paste for longer text
            return self.insert_text_paste(text)
        return self.insert_text_typing(text)
