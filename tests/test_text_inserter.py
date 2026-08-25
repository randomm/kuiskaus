"""Hardware-free unit tests for TextInserter (issue #41).

Quartz and AppKit are stubbed in sys.modules before the real
kuiskaus.text_inserter module is imported, mirroring tests/test_app.py's
_FakeAppKit pattern (text_inserter imports BOTH AppKit and Quartz at
module scope). CGEventPost is a MagicMock returning True by default; the
failure tests flip its return value per test.
"""

import importlib
import sys
from types import ModuleType
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
    # True by default: failure tests flip this per test.
    quartz.CGEventPost = MagicMock(return_value=True)
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

    return importlib.reload(ti).TextInserter()


@pytest.fixture
def quartz(monkeypatch: pytest.MonkeyPatch) -> _FakeQuartz:
    """The Quartz stub the current text_inserter module is bound to."""
    import kuiskaus.text_inserter as ti

    return ti.Quartz


@pytest.fixture
def pasteboard(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The NSPasteboard stub the current text_inserter module is bound to."""
    import kuiskaus.text_inserter as ti

    pb = MagicMock(name="pasteboard")
    ti.NSPasteboard.generalPasteboard.return_value = pb
    return pb


def _set_strings_called(pasteboard: MagicMock) -> list:
    return [c.args[0] for c in pasteboard.setString_forType_.call_args_list]


def test_insert_text_returns_true_on_success(inserter, quartz, pasteboard):
    """Happy path: every CGEventPost returns True, no error recorded."""
    result = inserter.insert_text("hello world")

    assert result is True
    assert inserter.last_error is None
    assert quartz.CGEventPost.call_count > 0
    assert inserter.insert_lock.locked() is False  # lock released


def test_insert_text_returns_false_on_cgeventpost_failure(inserter, quartz):
    """Any CGEventPost False surfaces False + a non-empty last_error."""
    quartz.CGEventPost.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert inserter.last_error is not None
    assert len(inserter.last_error) > 0


def test_insert_text_returns_false_on_pasteboard_failure(inserter, quartz, pasteboard):
    """NSPasteboard write False: no paste keystrokes, error recorded."""
    pasteboard.setString_forType_.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert inserter.last_error is not None
    # The paste simulation (Cmd+V) must NOT run when the clipboard
    # write failed.
    assert quartz.CGEventPost.call_count == 0


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


def test_simulate_paste_failure_reports_which_post_failed(
    inserter, quartz, monkeypatch
):
    """Each of the 4 Cmd+V posts, failing in turn, records a specific
    message and returns False (no TCC hint when called directly)."""
    # 1st False (Cmd-down), then all True: the V posts never run.
    quartz.CGEventPost.side_effect = [False, True, True, True]
    assert inserter._simulate_paste() is False
    assert "Cmd down" in inserter.last_error
    # 2nd False (V-down).
    quartz.CGEventPost.side_effect = [True, False, True, True]
    assert inserter._simulate_paste() is False
    assert "V down" in inserter.last_error
    # 3rd False (V-up).
    quartz.CGEventPost.side_effect = [True, True, False, True]
    assert inserter._simulate_paste() is False
    assert "V up" in inserter.last_error
    # 4th False (Cmd-up).
    quartz.CGEventPost.side_effect = [True, True, True, False]
    assert inserter._simulate_paste() is False
    assert "Cmd up" in inserter.last_error


def test_type_character_key_up_failure_reports_specifically(
    inserter, quartz, monkeypatch
):
    """Key-down posts, key-up fails: specific message, returns False."""
    quartz.CGEventPost.side_effect = [True, False]

    assert inserter._type_character("h") is False
    assert "key up" in inserter.last_error
    assert quartz.CGEventPost.call_count == 2


def test_insert_text_surfaces_tcc_revoked_on_axistrusted_false(
    inserter, quartz, monkeypatch
):
    """CGEventPost fails AND AXIsProcessTrusted()==False: revocation hint."""
    quartz.CGEventPost.return_value = False
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert "Accessibility permission revoked" in inserter.last_error
    assert app_services.AXIsProcessTrusted.call_count >= 1


def test_insert_text_surfaces_tahoe_hint_on_axistrusted_true(
    inserter, quartz, monkeypatch
):
    """CGEventPost fails but AXIsProcessTrusted()==True: Tahoe hint."""
    quartz.CGEventPost.return_value = False
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = True

    result = inserter.insert_text("hello world")

    assert result is False
    assert "Input Injection AND Accessibility" in inserter.last_error


def test_axistrusted_checked_once_per_session(inserter, quartz, monkeypatch):
    """Trust is cached after first failure; later failures reuse it."""
    quartz.CGEventPost.return_value = False
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    inserter.insert_text("first failure")
    inserter.insert_text("second failure")

    assert app_services.AXIsProcessTrusted.call_count == 1


def test_insert_text_typing_returns_false_on_per_char_failure(inserter, monkeypatch):
    """First _type_character False: loop breaks, returns False."""
    calls = {"n": 0}

    def fake_type_character(char):
        calls["n"] += 1
        return calls["n"] == 2  # fail on the 2nd character

    with patch.object(inserter, "_type_character", side_effect=fake_type_character):
        result = inserter.insert_text_typing("abcdef")

    assert result is False
    assert calls["n"] == 1  # loop broke immediately on the failing char
    assert inserter.last_error is not None
    # The per-char message is replaced by the TCC hint (see below).
    assert "Keyboard event injection failed" not in inserter.last_error


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


def test_insert_text_empty_text_returns_true_without_side_effects(inserter, quartz):
    """Empty text: early return True, nothing typed, no error."""
    result = inserter.insert_text("")

    assert result is True
    assert inserter.last_error is None
    quartz.CGEventPost.assert_not_called()


def test_last_error_cleared_at_start_of_each_insert_text(inserter, quartz, monkeypatch):
    """A failure's last_error is cleared at the START of the next call,
    before the new call's own failure message is written."""
    quartz.CGEventPost.return_value = False
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False  # revoked hint
    assert inserter.insert_text("first failure") is False
    first_error = inserter.last_error
    assert first_error is not None

    # A second failing call starts from a cleared last_error; the
    # assertion runs mid-call, right after the clear, before the new
    # failure message replaces it.
    def fail_at_second_call(text):
        assert inserter.last_error is None  # cleared at call start
        quartz.CGEventPost.return_value = False
        raise RuntimeError("fail here")

    with (
        patch.object(inserter, "insert_text_paste", side_effect=fail_at_second_call),
        pytest.raises(RuntimeError, match="fail here"),
    ):
        inserter.insert_text("second failure")


def test_short_text_uses_typing_path(inserter, quartz):
    """≤10 chars: typed character-by-character (2 posts per char)."""
    result = inserter.insert_text("hi")

    assert result is True
    assert quartz.CGEventPost.call_count == 4  # down + up per char
    assert quartz.CGEventPost.call_args_list[0].args[0] == (quartz.kCGSessionEventTap)
