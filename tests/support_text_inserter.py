"""Shared helpers for the test_text_inserter test modules (issue #70).

Hardware-free unit tests for TextInserter (issue #41, reworked for
#58).

AppKit is stubbed in sys.modules before the real kuiskaus.text_inserter
module is imported, mirroring tests/test_app.py's _FakeAppKit pattern.
osascript is the sole insertion path: exactly one osascript call per
insertion (issue #58 removed the CGEventPost path — PyObjC declares
CGEventPost void — and osascript is the reliable one on macOS 26
Tahoe). The production module does not import Quartz.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_PASTEBOARD_TYPE = "public.utf8-plain-text"


class _FakeAppKit(ModuleType):
    """AppKit stub. NSApp et al. are needed because kuiskaus/__init__.py
    imports hotkey_listener, which pulls in the REAL
    PyObjCTools.AppHelper and its module-scope `from AppKit import NSApp,
    ...` — none of these symbols are exercised here, placeholders only."""

    NSEvent: MagicMock
    NSPasteboard: MagicMock
    NSPasteboardTypeString: str
    NSApp: MagicMock
    NSApplicationDidFinishLaunchingNotification: str
    NSApplicationMain: MagicMock
    NSRunAlertPanel: MagicMock


class _FakeApplicationServices(ModuleType):
    AXIsProcessTrusted: MagicMock


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub AppKit before importing the real text_inserter."""
    appkit = _FakeAppKit("AppKit")
    appkit.NSEvent = MagicMock(name="NSEvent")
    appkit.NSPasteboard = MagicMock(name="NSPasteboard")
    appkit.NSPasteboardTypeString = _PASTEBOARD_TYPE
    appkit.NSApp = MagicMock(name="NSApp")
    appkit.NSApplicationDidFinishLaunchingNotification = "didFinishLaunching"
    appkit.NSApplicationMain = MagicMock(name="NSApplicationMain")
    appkit.NSRunAlertPanel = MagicMock(name="NSRunAlertPanel")
    monkeypatch.setitem(sys.modules, "AppKit", appkit)

    # ApplicationServices hosts AXIsProcessTrusted; text_inserter imports
    # it locally on the failure path, so the stub must carry the symbol.
    app_services = _FakeApplicationServices("ApplicationServices")
    app_services.AXIsProcessTrusted = MagicMock(return_value=True)
    monkeypatch.setitem(sys.modules, "ApplicationServices", app_services)


def _set_strings_called(pasteboard: MagicMock) -> list:
    return [c.args[0] for c in pasteboard.setString_forType_.call_args_list]


def _osascript_scripts(fake_run: MagicMock) -> list[str]:
    """The AppleScript payload passed to osascript for each call."""
    return [c.args[0][2] for c in fake_run.call_args_list]


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, fake_run: MagicMock) -> None:
    """Point the bound text_inserter module's subprocess.run at fake_run."""
    import kuiskaus.text_inserter as ti

    monkeypatch.setattr(ti.subprocess, "run", fake_run)


def _failing_osascript(stderr: str = "denied") -> MagicMock:
    """MagicMock for subprocess.run returning a failed osascript invocation."""
    return MagicMock(return_value=MagicMock(returncode=1, stderr=stderr))


def _ok_osascript() -> MagicMock:
    """MagicMock for subprocess.run returning a successful osascript run."""
    return MagicMock(return_value=MagicMock(returncode=0, stderr=""))
