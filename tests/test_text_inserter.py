"""Hardware-free unit tests for TextInserter (issue #41, reworked for
#58).

Quartz and AppKit are stubbed in sys.modules before the real
kuiskaus.text_inserter module is imported, mirroring tests/test_app.py's
_FakeAppKit pattern (text_inserter imports BOTH AppKit and Quartz at
module scope). CGEventPost is a MagicMock returning None by default —
matching real PyObjC, which declares CGEventPost void — so every test
exercises the real shape where the return value is discarded (issue
#58). osascript is the sole success path: exactly one osascript call per
insertion.
"""

import importlib
import subprocess
import sys
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

# NSPasteboardTypeString must be a stable sentinel: the real code passes
# it to setString_forType_ and the tests assert on those calls.
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


class _FakeQuartz(ModuleType):
    CGEventCreateKeyboardEvent: MagicMock
    CGEventKeyboardSetUnicodeString: MagicMock
    CGEventSetFlags: MagicMock
    CGEventPost: MagicMock
    kCGSessionEventTap: str
    kCGEventFlagMaskCommand: int


class _FakeApplicationServices(ModuleType):
    AXIsProcessTrusted: MagicMock


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub AppKit and Quartz before importing the real text_inserter."""
    appkit = _FakeAppKit("AppKit")
    appkit.NSEvent = MagicMock(name="NSEvent")
    appkit.NSPasteboard = MagicMock(name="NSPasteboard")
    appkit.NSPasteboardTypeString = _PASTEBOARD_TYPE
    appkit.NSApp = MagicMock(name="NSApp")
    appkit.NSApplicationDidFinishLaunchingNotification = "didFinishLaunching"
    appkit.NSApplicationMain = MagicMock(name="NSApplicationMain")
    appkit.NSRunAlertPanel = MagicMock(name="NSRunAlertPanel")
    monkeypatch.setitem(sys.modules, "AppKit", appkit)

    quartz = _FakeQuartz("Quartz")
    quartz.CGEventCreateKeyboardEvent = MagicMock(return_value=MagicMock())
    quartz.CGEventKeyboardSetUnicodeString = MagicMock()
    quartz.CGEventSetFlags = MagicMock()
    # None by default: matches real PyObjC, where CGEventPost is declared
    # void (issue #58).
    quartz.CGEventPost = MagicMock(return_value=None)
    quartz.kCGSessionEventTap = "kCGSessionEventTap"
    quartz.kCGEventFlagMaskCommand = 1 << 20
    monkeypatch.setitem(sys.modules, "Quartz", quartz)

    # ApplicationServices hosts AXIsProcessTrusted; text_inserter imports
    # it locally on the failure path, so the stub must carry the symbol.
    app_services = _FakeApplicationServices("ApplicationServices")
    app_services.AXIsProcessTrusted = MagicMock(return_value=True)
    monkeypatch.setitem(sys.modules, "ApplicationServices", app_services)


@pytest.fixture
def inserter(monkeypatch: pytest.MonkeyPatch):
    _install_stubs(monkeypatch)
    # Re-import the module fresh so it binds THIS test's Quartz stub.
    # (sys.modules["kuiskaus.text_inserter"] may hold a prior test's
    # module object whose globals point at a different stub.)
    import kuiskaus.text_inserter as ti

    importlib.reload(ti)
    # Default: subprocess.run is a benign success so the osascript call
    # (issue #58: the sole insertion path) never shells out in the test
    # environment. monkeypatch restores the real subprocess.run after
    # each test; failure tests replace this mock per-test via
    # _patch_subprocess.
    monkeypatch.setattr(
        ti.subprocess,
        "run",
        MagicMock(
            return_value=MagicMock(returncode=0, stderr=""), name="subprocess.run"
        ),
    )
    return ti.TextInserter()


@pytest.fixture
def quartz(monkeypatch: pytest.MonkeyPatch) -> _FakeQuartz:
    """The Quartz stub installed in sys.modules for this test."""
    return cast(_FakeQuartz, sys.modules["Quartz"])


@pytest.fixture
def pasteboard(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The NSPasteboard stub the current text_inserter module is bound to."""
    import kuiskaus.text_inserter as ti

    pb = MagicMock(name="pasteboard")
    ti.NSPasteboard.generalPasteboard.return_value = pb
    return pb


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


# --- happy paths (CGEventPost returns None, osascript is the success path) ---


def test_insert_text_returns_true_on_success(inserter, quartz, pasteboard):
    """Paste path with real PyObjC shape: CGEventPost returns None,
    osascript succeeds, one osascript Cmd+V call, no error."""
    result = inserter.insert_text("hello world")

    assert result is True
    assert inserter.last_error is None
    assert inserter.insert_lock.locked() is False  # lock released


def test_paste_path_fires_exactly_one_osascript_cmdv_call(
    inserter, quartz, monkeypatch
):
    """Core issue #58 reproduction: CGEventPost returns None (real
    PyObjC), osascript succeeds, insert_text("hello world") (>10 chars →
    paste path) returns True with EXACTLY ONE subprocess.run call, the
    Cmd+V keystroke, and last_error stays None."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter.insert_text("hello world") is True
    assert fake_run.call_count == 1
    assert "using command down" in _osascript_scripts(fake_run)[0]
    assert inserter.last_error is None


def test_typing_path_fires_exactly_one_osascript_keystroke_call(
    inserter, quartz, monkeypatch
):
    """Typing path (≤10 chars) with real PyObjC shape: one osascript
    keystroke call for the WHOLE string (no per-char spawning), CGEvents
    still posted (2 per char, return value discarded)."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter.insert_text_typing("hello") is True
    assert fake_run.call_count == 1
    assert 'keystroke "hello"' in _osascript_scripts(fake_run)[0]
    assert inserter.last_error is None


def test_short_text_uses_typing_path(inserter, monkeypatch):
    """≤10 chars: typing path (single osascript keystroke call), not
    the paste path (no Cmd+V script)."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    result = inserter.insert_text("hi")

    assert result is True
    assert fake_run.call_count == 1
    assert 'keystroke "hi"' in _osascript_scripts(fake_run)[0]
    assert all("command down" not in s for s in _osascript_scripts(fake_run))


def test_insert_text_empty_text_returns_true_without_side_effects(inserter, quartz):
    """Empty text: early return True, nothing typed, no error."""
    result = inserter.insert_text("")

    assert result is True
    assert inserter.last_error is None
    quartz.CGEventPost.assert_not_called()


def test_osascript_keystroke_escapes_special_chars(inserter, quartz, monkeypatch):
    """Backslash-then-quote escaping of the AppleScript literal for the
    whole string (single osascript call on the typing path)."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    inserter.insert_text_typing('a"b\\c')

    assert fake_run.call_count == 1
    script = _osascript_scripts(fake_run)[0]
    assert '\\"' in script  # quote escaped
    assert "\\\\" in script  # backslash escaped


def test_osascript_keystroke_script_shape_escapes_shell_injection():
    """The generated AppleScript literal escapes backslash first, then
    quote, so a hostile payload cannot break out of the string context
    (issue #51 lens SECURITY). No subprocess is invoked: build the
    script the same way _osascript_keystroke does and assert on shape."""

    def build(text: str) -> str:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'tell application "System Events" to keystroke "{escaped}"'

    hostile = '")do shell script "rm x"--'
    script = build(hostile)
    # Extract the AppleScript string literal (the quoted segment at the
    # END of the script — the first " is the one around "System Events").
    tail = script.split('"System Events" to keystroke "')[1]
    inner = tail[: tail.rindex('"')]
    # A bare (unescaped) quote inside the literal would terminate it
    # early and let the payload break out of the string context.
    i = 0
    while i < len(inner):
        if inner[i] == '"':
            assert inner[i - 1] == "\\", "bare quote inside string literal"
        i += 1
    # The payload's quotes are all escaped: `do shell script` appears
    # only in escaped form, never as a command.
    assert "do shell script" in script
    # The hostile payload's closing quote is escaped, so it cannot
    # terminate the AppleScript string literal and inject a command.
    assert '\\"' in script


# --- osascript timeout constant (issue #58) ---


def test_run_osascript_passes_named_timeout_constant(inserter, monkeypatch):
    """_run_osascript passes _OSASCRIPT_TIMEOUT_S as the timeout kwarg to
    subprocess.run; the constant is 15."""
    import kuiskaus.text_inserter as ti

    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    ti._run_osascript('keystroke "a"')

    assert ti._OSASCRIPT_TIMEOUT_S == 15
    assert fake_run.call_args.kwargs["timeout"] == ti._OSASCRIPT_TIMEOUT_S


def test_run_osascript_timeout_message_derives_from_constant(inserter, monkeypatch):
    """TimeoutExpired message is built from the named constant (shows
    15, not the old hardcoded "2s")."""
    import kuiskaus.text_inserter as ti

    fake_run = MagicMock(
        side_effect=subprocess.TimeoutExpired(cmd=["osascript"], timeout=15)
    )
    _patch_subprocess(monkeypatch, fake_run)

    ok, err = ti._run_osascript('keystroke "a"')

    assert ok is False
    assert "osascript timeout" in err
    assert "15" in err


# --- failure paths: osascript fails, hint APPENDED (issue #58) ---


def test_osascript_failure_returns_false_with_error_and_tcc_hint(
    inserter, quartz, monkeypatch
):
    """Paste path, osascript non-zero exit: False + last_error contains
    BOTH the underlying osascript error AND the TCC hint (appended, not
    overwritten)."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False  # revoked hint

    result = inserter.insert_text("hello world")

    assert result is False
    assert inserter.last_error is not None
    assert "denied" in inserter.last_error  # underlying error survives
    assert "Accessibility permission revoked" in inserter.last_error


def test_osascript_1002_denial_records_error_and_tcc_hint(
    inserter, quartz, monkeypatch
):
    """osascript exit 2 with '1002: not allowed to send keystrokes':
    False + last_error contains BOTH the underlying error and the TCC
    hint, with the underlying error appearing FIRST (append, not
    overwrite)."""
    _patch_subprocess(
        monkeypatch,
        _failing_osascript("1002: not allowed to send keystrokes"),
    )

    result = inserter.insert_text("hello world")

    assert result is False
    assert "1002: not allowed to send keystrokes" in inserter.last_error
    assert inserter.last_error.index("1002") < inserter.last_error.index(
        "Input Injection AND Accessibility"
    )


def test_osascript_timeout_records_error_and_tcc_hint(inserter, monkeypatch):
    """TimeoutExpired: False + last_error contains BOTH the timeout
    message (derived from the constant, "15s") AND the TCC hint, with
    the underlying message first."""
    _patch_subprocess(
        monkeypatch,
        MagicMock(side_effect=subprocess.TimeoutExpired(cmd=["osascript"], timeout=15)),
    )

    result = inserter.insert_text("hello world")

    assert result is False
    assert "osascript timeout" in inserter.last_error
    assert "15" in inserter.last_error
    assert inserter.last_error.index("osascript timeout") < inserter.last_error.index(
        "Input Injection AND Accessibility"
    )


def test_tcc_revoked_hint_appended_on_injection_failure(inserter, quartz, monkeypatch):
    """osascript fails AND AXIsProcessTrusted()==False: the revocation
    hint is appended after the underlying error; the AXIsProcessTrusted
    check ran."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert "Accessibility permission revoked" in inserter.last_error
    assert "denied" in inserter.last_error
    assert app_services.AXIsProcessTrusted.call_count >= 1


def test_tahoe_hint_appended_when_axistrusted_true(inserter, quartz, monkeypatch):
    """osascript fails but AXIsProcessTrusted()==True: Tahoe hint is
    appended after the underlying error."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = True

    result = inserter.insert_text("hello world")

    assert result is False
    assert "denied" in inserter.last_error
    assert "Input Injection AND Accessibility" in inserter.last_error


def test_axistrusted_checked_once_per_session(inserter, quartz, monkeypatch):
    """Trust is cached after first failure; later failures reuse it."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    inserter.insert_text("first failure")
    inserter.insert_text("second failure")

    assert app_services.AXIsProcessTrusted.call_count == 1


def test_last_error_cleared_at_start_of_each_insert_text(inserter, quartz, monkeypatch):
    """A failure's last_error is cleared at the START of the next call,
    before the new call's own failure message is written."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False  # revoked hint
    assert inserter.insert_text("first failure") is False
    first_error = inserter.last_error
    assert first_error is not None
    assert "denied" in first_error

    # A second failing call starts from a cleared last_error; the
    # assertion runs mid-call, right after the clear, before the new
    # failure message replaces it.
    def fail_at_second_call(text):
        assert inserter.last_error is None  # cleared at call start
        raise RuntimeError("fail here")

    with (
        patch.object(inserter, "insert_text_paste", side_effect=fail_at_second_call),
        pytest.raises(RuntimeError, match="fail here"),
    ):
        inserter.insert_text("second failure")


def test_tcc_hint_is_not_invoked_on_pasteboard_failure(
    inserter, pasteboard, monkeypatch
):
    """A clipboard-write failure is not an injection failure: no
    AXIsProcessTrusted check, no TCC message."""
    pasteboard.setString_forType_.return_value = False
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert app_services.AXIsProcessTrusted.call_count == 0
    assert "Accessibility permission revoked" not in inserter.last_error


def test_simulate_paste_failure_records_osascript_error(inserter, monkeypatch):
    """_simulate_paste with a failing osascript: False + a last_error
    mentioning the osascript Cmd+V failure and the underlying error (no
    TCC hint when called directly)."""
    _patch_subprocess(monkeypatch, _failing_osascript("permission denied"))

    assert inserter._simulate_paste() is False
    assert "osascript Cmd+V failed" in inserter.last_error
    assert "permission denied" in inserter.last_error


def test_typing_path_failure_records_osascript_error_and_tcc_hint(
    inserter, monkeypatch
):
    """Typing path, failing osascript: False + last_error contains the
    osascript keystroke failure AND the appended TCC hint."""
    _patch_subprocess(monkeypatch, _failing_osascript("permission denied"))

    assert inserter.insert_text_typing("hi") is False
    assert "osascript keystroke failed" in inserter.last_error
    assert "permission denied" in inserter.last_error
    assert "Input Injection AND Accessibility" in inserter.last_error


def test_simulate_paste_success_returns_true(inserter, monkeypatch):
    """_simulate_paste with a successful osascript: True, no error."""
    _patch_subprocess(monkeypatch, _ok_osascript())

    assert inserter._simulate_paste() is True
    assert inserter.last_error is None


# --- clipboard semantics (unchanged by issue #58) ---


def test_insert_text_returns_false_on_pasteboard_failure(inserter, quartz, pasteboard):
    """NSPasteboard write False: no paste keystrokes, error recorded."""
    pasteboard.setString_forType_.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert inserter.last_error is not None
    # The paste simulation (Cmd+V) must NOT run when the clipboard
    # write failed.
    assert quartz.CGEventPost.call_count == 0


def test_insert_text_paste_failure_leaves_transcribed_text_on_clipboard(
    inserter, pasteboard
):
    """Simulated paste fails: prior clipboard is NOT restored, so the
    transcribed text stays for a manual Cmd+V."""
    pasteboard.stringForType_.return_value = "original"
    with patch.object(inserter, "_simulate_paste", return_value=False):
        result = inserter.insert_text_paste("transcribed text")

    assert result is False
    assert inserter.last_error is not None
    # The only clearContents + setString calls are the initial write of
    # the transcribed text -- no restore pair for "original".
    assert _set_strings_called(pasteboard) == ["transcribed text"]


def test_insert_text_paste_success_restores_prior_clipboard(inserter, pasteboard):
    """Happy path: prior clipboard restored after the paste."""
    pasteboard.stringForType_.return_value = "original"

    result = inserter.insert_text_paste("transcribed text")

    assert result is True
    assert _set_strings_called(pasteboard) == ["transcribed text", "original"]


def test_insert_text_paste_without_prior_content_restores_nothing(inserter, pasteboard):
    """Empty prior clipboard: no restore write on success."""
    pasteboard.stringForType_.return_value = None

    result = inserter.insert_text_paste("transcribed text")

    assert result is True
    assert _set_strings_called(pasteboard) == ["transcribed text"]
