import subprocess
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


def _run_osascript(script: str) -> tuple[bool, str]:
    """Run an AppleScript via osascript. Returns (ok, error detail).

    2s timeout: keystroke scripts complete in <100ms, but the first
    invocation in a session can cold-launch System Events, which stays
    well under 2s on a healthy machine.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        return (
            False,
            (result.stderr or "").strip() or f"exit code {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return False, "osascript timeout (2s)"
    except Exception as e:  # noqa: BLE001 - subprocess boundary
        return False, f"osascript subprocess error: {e}"


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
        # Once CGEventPost has failed this session, per-char CGEvent
        # attempts are doomed: remember it so future insertions batch
        # straight to osascript instead of N per-char subprocess spawns
        # (issue #51 lens PERFORMANCE).
        self._cgevent_broken: bool = False

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

            # Bypass: CGEvent known broken this session — one batched
            # osascript call instead of N per-char subprocess spawns.
            if self._cgevent_broken:
                return self._fallback_keystroke_batch(text)

            for i, char in enumerate(text):
                if not self._type_character(char):
                    self._surface_tcc_hint()
                    return False
                if self._cgevent_broken:
                    # _type_character just fell back to osascript (it
                    # remembered the breakage): batch the remaining
                    # characters in a single osascript call instead of
                    # N more per-char subprocess spawns.
                    remaining = text[i + 1 :]
                    if remaining:
                        return self._fallback_keystroke_batch(remaining)
                    return True
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
        """Type a single character. Returns False if both CGEventPost
        AND the osascript fallback failed (issue #51)."""
        # Create key down event
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, event):
            return self._fallback_keystroke("key down", char)

        # Create key up event
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(char), char)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, event):
            return self._fallback_keystroke("key up", char)
        return True

    def _simulate_paste(self) -> bool:
        """Simulate Cmd+V key combination. Returns False if both
        CGEventPost AND the osascript fallback failed (issue #51)."""
        # Bypass: CGEvent known broken this session — osascript Cmd+V
        # directly, no doomed CGEventPost attempts.
        if self._cgevent_broken:
            return self._fallback_cmd_v("Cmd down (cgevent broken)")

        # Key code for 'v' is 9
        v_key = 9

        def _fallback(step: str) -> bool:
            ok = self._fallback_cmd_v(step)
            # CGEvent broke and the osascript fallback worked: remember
            # it so the next paste skips doomed CGEvent attempts.
            if ok:
                self._cgevent_broken = True
            return ok

        # Press Cmd key
        cmd_down = Quartz.CGEventCreateKeyboardEvent(
            None, 0x37, True
        )  # 0x37 is Command key
        Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_down):
            return _fallback("Cmd down")

        # Press 'v' with Cmd held
        v_down = Quartz.CGEventCreateKeyboardEvent(None, v_key, True)
        Quartz.CGEventSetFlags(v_down, Quartz.kCGEventFlagMaskCommand)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_down):
            return _fallback("V down")

        # Release 'v'
        v_up = Quartz.CGEventCreateKeyboardEvent(None, v_key, False)
        Quartz.CGEventSetFlags(v_up, Quartz.kCGEventFlagMaskCommand)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, v_up):
            return _fallback("V up")

        # Release Cmd
        cmd_up = Quartz.CGEventCreateKeyboardEvent(None, 0x37, False)
        if not Quartz.CGEventPost(Quartz.kCGSessionEventTap, cmd_up):
            return _fallback("Cmd up")
        return True

    def _fallback_keystroke_batch(self, text: str) -> bool:
        """Type the WHOLE string in one osascript call (issue #51 lens
        PERFORMANCE): per-char subprocess spawning blocks the UI for
        N * ~100-300ms on the fallback path. Error reporting matches
        _fallback_keystroke."""
        ok, err = self._osascript_keystroke(text)
        if ok:
            return True
        self.last_error = (
            f"CGEventPost is broken this session AND osascript batch "
            f"keystroke failed: {err}"
        )
        return False

    def _fallback_keystroke(self, cgevent_step: str, char: str) -> bool:
        """Fall back to osascript when CGEventPost(cgevent_step) failed.

        osascript is Apple-signed with a stable TCC identity, so its
        automation grant survives uv-Python path/signature churn that
        breaks CGEventPost on macOS 26 (issue #51). On success, remember
        that CGEvent is broken this session so the next insertion can
        batch straight to osascript (issue #51 lens PERFORMANCE).
        """
        ok, err = self._osascript_keystroke(char)
        if ok:
            self._cgevent_broken = True
            return True
        self.last_error = (
            f"CGEventPost failed ({cgevent_step}) AND osascript keystroke failed: {err}"
        )
        return False

    def _fallback_cmd_v(self, cgevent_step: str) -> bool:
        """Fall back to osascript Cmd+V when CGEventPost(cgevent_step)
        failed. The pasteboard was already populated by the caller, so
        this only needs to trigger the paste keystroke."""
        ok, err = self._osascript_cmd_v()
        if ok:
            return True
        self.last_error = (
            f"CGEventPost failed ({cgevent_step}) AND osascript Cmd+V failed: {err}"
        )
        return False

    @staticmethod
    def _osascript_keystroke(text: str) -> tuple[bool, str]:
        """Type text via osascript keystroke. Returns (ok, error).

        Escapes backslash first, then double-quote (order matters) for
        the AppleScript string literal.
        """
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        return _run_osascript(script)

    @staticmethod
    def _osascript_cmd_v() -> tuple[bool, str]:
        """Trigger Cmd+V via osascript. Returns (ok, error)."""
        script = 'tell application "System Events" to keystroke "v" using command down'
        return _run_osascript(script)

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
