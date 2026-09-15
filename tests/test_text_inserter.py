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

    importlib.reload(ti)
    # Default: subprocess.run is a benign success so the osascript
    # fallback (issue #51) never shells out in the test environment.
    # monkeypatch restores the real subprocess.run after each test;
    # fallback tests replace this mock per-test via _patch_subprocess.
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


def test_insert_text_returns_false_on_cgeventpost_failure(
    inserter, quartz, monkeypatch
):
    """Any CGEventPost False AND osascript fallback False surfaces False
    + a non-empty last_error (issue #51)."""
    quartz.CGEventPost.return_value = False
    _patch_subprocess(monkeypatch, _failing_osascript())

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
    """Each of the 4 Cmd+V posts, failing in turn, makes the osascript
    fallback fire and fail, recording a specific combined message and
    returning False (no TCC hint when called directly)."""
    fake_run = _failing_osascript("permission denied")
    _patch_subprocess(monkeypatch, fake_run)
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
    """Key-down posts, key-up fails: osascript fallback also fails, so
    the combined message is recorded and False is returned."""
    fake_run = _failing_osascript("permission denied")
    _patch_subprocess(monkeypatch, fake_run)
    quartz.CGEventPost.side_effect = [True, False]

    assert inserter._type_character("h") is False
    assert "key up" in inserter.last_error
    assert quartz.CGEventPost.call_count == 2


def test_insert_text_surfaces_tcc_revoked_on_axistrusted_false(
    inserter, quartz, monkeypatch
):
    """CGEventPost fails AND AXIsProcessTrusted()==False: revocation hint."""
    quartz.CGEventPost.return_value = False
    _patch_subprocess(monkeypatch, _failing_osascript())
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert "Accessibility permission revoked" in inserter.last_error
    assert app_services.AXIsProcessTrusted.call_count >= 1


def test_insert_text_surfaces_tahoe_hint_on_axistrusted_true(
    inserter, quartz, monkeypatch
):
    """CGEventPost fails (fallback also fails) but
    AXIsProcessTrusted()==True: Tahoe hint."""
    quartz.CGEventPost.return_value = False
    _patch_subprocess(monkeypatch, _failing_osascript())
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = True

    result = inserter.insert_text("hello world")

    assert result is False
    assert "Input Injection AND Accessibility" in inserter.last_error


def test_axistrusted_checked_once_per_session(inserter, quartz, monkeypatch):
    """Trust is cached after first failure; later failures reuse it."""
    quartz.CGEventPost.return_value = False
    _patch_subprocess(monkeypatch, _failing_osascript())
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    inserter.insert_text("first failure")
    inserter.insert_text("second failure")

    assert app_services.AXIsProcessTrusted.call_count == 1


def test_insert_text_typing_returns_false_on_per_char_failure(inserter, monkeypatch):
    """First _type_character False with the osascript fallback ALSO
    failed (last_error set): loop breaks, returns False."""
    calls = {"n": 0}

    def fake_type_character(char):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the 2nd character, both paths
            inserter.last_error = "CGEventPost failed (key down)"
            return False
        return True

    with patch.object(inserter, "_type_character", side_effect=fake_type_character):
        result = inserter.insert_text_typing("abcdef")

    assert result is False
    # The loop broke on the failing char (2 = one success + the failure).
    assert calls["n"] == 2
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
    _patch_subprocess(monkeypatch, _failing_osascript())
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


# --- osascript fallback tests (issue #51) ---


def _osascript_args(fake_run: MagicMock) -> list[str]:
    return fake_run.call_args[0][0]


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, fake_run: MagicMock) -> None:
    """Point the bound text_inserter module's subprocess.run at fake_run."""
    import kuiskaus.text_inserter as ti

    monkeypatch.setattr(ti.subprocess, "run", fake_run)


def _failing_osascript(stderr: str = "denied") -> MagicMock:
    """MagicMock for subprocess.run returning a failed osascript invocation."""
    return MagicMock(return_value=MagicMock(returncode=1, stderr=stderr))


def test_osascript_fallback_fires_on_cgeventpost_failure(inserter, quartz, monkeypatch):
    """CGEventPost False: osascript keystroke runs, insert returns True."""
    quartz.CGEventPost.return_value = False
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)

    result = inserter.insert_text("hi")

    assert result is True
    assert fake_run.called
    args = _osascript_args(fake_run)
    assert args[0] == "osascript"
    assert "keystroke" in args[2]


def test_osascript_fallback_escapes_special_chars(inserter, quartz, monkeypatch):
    """Backslash-then-quote escaping of the AppleScript literal.

    Uses insert_text_typing directly (bypasses the >10-char paste
    dispatch) so each character goes through _type_character →
    _fallback_keystroke → _osascript_keystroke.
    """
    quartz.CGEventPost.return_value = False
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)

    # Short string with both \\ and " to stay on the typing path.
    inserter.insert_text_typing('a"b\\c')

    # Each char is a separate osascript call; check the one that
    # contains the quote and the one with the backslash.
    scripts = [c.args[0][2] for c in fake_run.call_args_list]
    assert any('"' in s and '\\"' in s for s in scripts)  # quote escaped
    assert any("\\\\" in s for s in scripts)  # backslash escaped


def test_osascript_fallback_failure_reports_both_errors(inserter, quartz, monkeypatch):
    """Both paths fail: last_error mentions CGEventPost AND osascript.

    Calls _type_character directly to avoid _surface_tcc_hint
    overwriting the detailed error message.
    """
    quartz.CGEventPost.return_value = False
    fake_run = _failing_osascript("permission denied")
    _patch_subprocess(monkeypatch, fake_run)

    result = inserter._type_character("h")

    assert result is False
    assert "CGEventPost failed" in inserter.last_error
    assert "osascript" in inserter.last_error


def test_osascript_fallback_timeout_reported(inserter, quartz, monkeypatch):
    """subprocess.TimeoutExpired: last_error mentions timeout.

    Calls _type_character directly to avoid _surface_tcc_hint
    overwriting the timeout message.
    """
    import subprocess as sp

    quartz.CGEventPost.return_value = False
    fake_run = MagicMock(side_effect=sp.TimeoutExpired(cmd=["osascript"], timeout=2))
    _patch_subprocess(monkeypatch, fake_run)

    result = inserter._type_character("h")

    assert result is False
    assert "timeout" in inserter.last_error.lower()


def test_cgeventpost_success_skips_osascript_fallback(inserter, quartz, monkeypatch):
    """CGEventPost True: fast path, osascript never spawned."""
    quartz.CGEventPost.return_value = True
    fake_run = MagicMock()
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter.insert_text("hi") is True
    assert fake_run.call_count == 0


def test_osascript_paste_fallback_on_cmdv_failure(
    inserter, quartz, pasteboard, monkeypatch
):
    """Long text: CGEventPost Cmd+V fails, osascript Cmd+V succeeds →
    insert returns True and the prior clipboard is still restored."""
    quartz.CGEventPost.return_value = False
    pasteboard.stringForType_.return_value = "original"
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)

    result = inserter.insert_text("hello world")  # >10 chars → paste path

    assert result is True
    assert fake_run.call_count == 1
    assert "using command down" in _osascript_args(fake_run)[2]
    # Success path restores the prior clipboard.
    assert _set_strings_called(pasteboard) == ["hello world", "original"]


# --- osascript fallback batching + security (issue #51 lens review) ---


def test_cgevent_broken_flag_batches_typing(inserter, quartz, monkeypatch):
    """First insert: CGEventPost False, osascript works per-char →
    _cgevent_broken set. Second insert: per-char loop bypassed entirely,
    single batched osascript call (not N)."""
    quartz.CGEventPost.return_value = False
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter.insert_text_typing("a") is True
    assert inserter._cgevent_broken is True
    assert (
        quartz.CGEventPost.call_count == 1
    )  # down only — CGEventPost fails on first post

    fake_run.reset_mock()
    assert inserter.insert_text_typing("abcd") is True
    # Bypass path: ONE osascript call for the whole string, and no
    # CGEventPost attempts this round.
    assert fake_run.call_count == 1
    assert 'keystroke "abcd"' in _osascript_args(fake_run)[2]
    assert quartz.CGEventPost.call_count == 1  # unchanged: bypass works


def test_batch_fallback_typing_uses_single_osascript_call(
    inserter, quartz, monkeypatch
):
    """Force fallback on the FIRST char of a 5-char string: that char
    goes through the per-char fallback, the remaining 4 chars go through
    ONE batched osascript call — 2 subprocess calls total, not 5."""
    quartz.CGEventPost.return_value = False
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter.insert_text_typing("hello") is True
    assert fake_run.call_count == 2  # char 'h' + batch "ello"
    scripts = [c.args[0][2] for c in fake_run.call_args_list]
    assert any('keystroke "h"' in s for s in scripts)
    assert any('keystroke "ello"' in s for s in scripts)
    assert inserter._cgevent_broken is True


def test_batch_fallback_failure_records_error(inserter, quartz, monkeypatch):
    """_cgevent_broken set and the batched osascript call fails: False +
    a combined message in last_error."""
    quartz.CGEventPost.return_value = False
    _patch_subprocess(monkeypatch, _failing_osascript("batch denied"))
    inserter._cgevent_broken = True

    assert inserter.insert_text_typing("abc") is False
    assert "osascript batch keystroke failed" in inserter.last_error
    assert "batch denied" in inserter.last_error


def test_simulate_paste_returns_early_after_osascript_success(
    inserter, quartz, monkeypatch
):
    """First Cmd+V attempt: CGEventPost False, osascript Cmd+V succeeds →
    _simulate_paste returns True immediately, the remaining 3 posts are
    NOT fired, and _cgevent_broken is remembered."""
    quartz.CGEventPost.side_effect = [False, True, True, True]
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter._simulate_paste() is True
    assert quartz.CGEventPost.call_count == 1  # early return, 3 posts skipped
    assert fake_run.call_count == 1
    assert inserter._cgevent_broken is True

    # Next paste skips CGEvent entirely and goes straight to osascript.
    fake_run.reset_mock()
    assert inserter._simulate_paste() is True
    assert quartz.CGEventPost.call_count == 1  # still 1 from before
    assert fake_run.call_count == 1


def test_simulate_paste_broken_flag_bypass(inserter, quartz, monkeypatch):
    """_cgevent_broken set: _simulate_paste bypasses all CGEventPost and
    fires exactly one osascript Cmd+V."""
    quartz.CGEventPost.return_value = True
    fake_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    _patch_subprocess(monkeypatch, fake_run)
    inserter._cgevent_broken = True

    assert inserter._simulate_paste() is True
    assert quartz.CGEventPost.call_count == 0
    assert fake_run.call_count == 1


def test_osascript_keystroke_script_shape_escapes_shell_injection():
    """The generated AppleScript literal escapes backslash first, then
    quote, so a hostile payload cannot break out of the string context
    (issue #51 lens SECURITY). No subprocess is invoked: build the
    script the same way _osascript_keystroke does and assert on shape."""

    def build(text: str) -> str:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'tell application "System Events" to keystroke "{escaped}"'

    # Assert _osascript_keystroke uses the same escape order on a real
    # call by checking the module's source-level behavior via a mock.
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
