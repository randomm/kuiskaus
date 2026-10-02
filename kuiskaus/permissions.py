"""macOS privacy-permission checks (issue #64).

All checks are silent (no TCC prompt). Grants attach to the process that
launched Kuiskaus (the terminal app), so these report that host's state.
"""

import subprocess

import AVFoundation
import Quartz
from ApplicationServices import AXIsProcessTrusted

# AVAuthorizationStatus values.
_AV_NOT_DETERMINED = 0
_AV_AUTHORIZED = 3

_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?{pane}"
_PANES = {
    "Microphone": "Privacy_Microphone",
    "Input Monitoring": "Privacy_ListenEvent",
    "Accessibility": "Privacy_Accessibility",
}


def _microphone_status() -> int:
    return AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeAudio
    )


def missing_permissions() -> list[str]:
    """Names of required permissions not currently granted, in a stable
    order (Microphone, Input Monitoring, Accessibility)."""
    missing = []
    if _microphone_status() != _AV_AUTHORIZED:
        missing.append("Microphone")
    if not Quartz.CGPreflightListenEventAccess():
        missing.append("Input Monitoring")
    if not AXIsProcessTrusted():
        missing.append("Accessibility")
    return missing


def request_microphone() -> None:
    """Trigger the system microphone prompt, only when never asked. Doing
    this up front keeps the prompt from landing mid-recording, where it
    blocks the stream open."""
    if _microphone_status() == _AV_NOT_DETERMINED:
        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeAudio, lambda _granted: None
        )


def open_settings(name: str) -> None:
    """Open the System Settings pane for a permission name; unknown names
    are ignored."""
    pane = _PANES.get(name)
    if pane is not None:
        subprocess.run(["open", _SETTINGS_URL.format(pane=pane)], check=False)


def status_title() -> str:
    """Menu text: which permissions are missing, or that all are granted."""
    missing = missing_permissions()
    if not missing:
        return "✅ Permissions granted"
    return f"⚠️ Grant: {', '.join(missing)} (click to open)"


def open_first_missing(_sender=None) -> None:
    """Menu click: open the Settings pane of the first missing permission."""
    missing = missing_permissions()
    if missing:
        open_settings(missing[0])
