import subprocess
import threading
import time

from AppKit import NSPasteboard, NSPasteboardTypeString

# System Events cold-start can block the osascript call for ~10s (issue
# #58 operator measurement: 10.2s wall-clock at 0.05s CPU, waiting on
# System Events/TCC), so the timeout must comfortably exceed that.
_OSASCRIPT_TIMEOUT_S = 15

# TCC hint messages. AXIsProcessTrusted is a hint, not a detector: on
# macOS 26 Tahoe, osascript keystrokes can fail even when the process is
# marked trusted, because Input Injection and Accessibility are tracked
# separately — so the trusted-case message steers the user to check
# both grants rather than assume the grants are fine.
_TCC_REVOKED_MESSAGE = (
    "Accessibility permission revoked — re-grant in System Settings > "
    "Privacy & Security > Accessibility (then restart the app)"
)
_TCC_TAHOE_MESSAGE = (
    "Text insertion failed — verify Input Injection AND Accessibility "
    "grants (macOS 26 tracks these separately in System Settings > "
    "Privacy & Security). If both are granted, this may be a Tahoe "
    "silent-drop bug."
)


def _run_osascript(script: str) -> tuple[bool, str]:
    """Run an AppleScript via osascript. Returns (ok, error detail).

    _OSASCRIPT_TIMEOUT_S: keystroke scripts complete in <100ms, but the
    first invocation in a session can cold-launch System Events, which
    can block for ~10s (issue #58).
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_OSASCRIPT_TIMEOUT_S,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        return (
            False,
            (result.stderr or "").strip() or f"exit code {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return False, f"osascript timeout ({_OSASCRIPT_TIMEOUT_S}s)"
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

    def insert_text_typing(self, text: str) -> bool:
        """
        Insert text by simulating keyboard typing

        Types the whole string in exactly one osascript keystroke call.
        osascript is the sole insertion path (issue #58 removed the
        CGEventPost path — PyObjC's CGEventPost is void, so its failure
        cannot even be detected), and on macOS 26 Tahoe it is the
        reliable one.

        Args:
            text: Text to insert

        Returns:
            False if the osascript keystroke failed (see last_error),
            else True.
        """
        with self.insert_lock:
            # Small delay to ensure we're ready
            time.sleep(0.1)

            ok, err = self._osascript_keystroke(text)
            if ok:
                return True
            self.last_error = f"osascript keystroke failed: {err}"
            self._surface_tcc_hint()
            return False

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

    def _simulate_paste(self) -> bool:
        """Simulate Cmd+V via a single osascript call. Returns False if
        the osascript Cmd+V failed (see last_error)."""
        ok, err = self._osascript_cmd_v()
        if ok:
            return True
        self.last_error = f"osascript Cmd+V failed: {err}"
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
        """Append a TCC hint to last_error after an injection failure.

        APPENDS (underlying error first, then the hint) so the real
        osascript error survives (issue #58 — the previous implementation
        overwrote last_error, hiding it). AXIsProcessTrusted() is cached
        for the session (trust, once revoked, only changes on re-grant +
        restart). The result is a hint for the message text, not a
        detector — see the module comment on the two message constants.
        """
        if self._ax_trusted is None:
            from ApplicationServices import AXIsProcessTrusted

            self._ax_trusted = bool(AXIsProcessTrusted())
        hint = _TCC_REVOKED_MESSAGE if not self._ax_trusted else _TCC_TAHOE_MESSAGE
        if self.last_error:
            self.last_error = f"{self.last_error} — {hint}"
        else:
            self.last_error = hint

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
