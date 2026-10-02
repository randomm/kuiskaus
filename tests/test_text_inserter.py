"""test_text_inserter tests, part 1 of 2 (issue #70)."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.support_text_inserter import (
    _failing_osascript,
    _ok_osascript,
    _osascript_scripts,
    _patch_subprocess,
)


# --- happy paths (CGEventPost returns None, osascript is the success path) ---
def test_module_does_not_import_quartz():
    """Guards that the CGEventPost-era Quartz import stayed removed
    (issue #58: osascript is the sole insertion path)."""
    import kuiskaus.text_inserter

    assert "Quartz" not in vars(kuiskaus.text_inserter)


def test_insert_text_returns_true_on_success(inserter, pasteboard):
    """Paste path with real PyObjC shape: osascript succeeds, one
    osascript Cmd+V call, no error."""
    result = inserter.insert_text("hello world")

    assert result is True
    assert inserter.last_error is None
    assert inserter.insert_lock.locked() is False  # lock released


def test_paste_path_fires_exactly_one_osascript_cmdv_call(inserter, monkeypatch):
    """Core issue #58 reproduction: osascript succeeds, insert_text(
    "hello world") (>10 chars → paste path) returns True with EXACTLY
    ONE subprocess.run call, the Cmd+V keystroke, and last_error stays
    None."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    assert inserter.insert_text("hello world") is True
    assert fake_run.call_count == 1
    assert "using command down" in _osascript_scripts(fake_run)[0]
    assert inserter.last_error is None


def test_typing_path_fires_exactly_one_osascript_keystroke_call(inserter, monkeypatch):
    """Typing path (≤10 chars): one osascript keystroke call for the
    WHOLE string (no per-char spawning)."""
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


def test_insert_text_empty_text_returns_true_without_side_effects(
    inserter, monkeypatch
):
    """Empty text: early return True, nothing typed (no osascript
    call), no error."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    result = inserter.insert_text("")

    assert result is True
    assert inserter.last_error is None
    assert fake_run.call_count == 0


def test_osascript_keystroke_escapes_special_chars(inserter, monkeypatch):
    """Backslash-then-quote escaping of the AppleScript literal for the
    whole string (single osascript call on the typing path)."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    inserter.insert_text_typing('a"b\\c')

    assert fake_run.call_count == 1
    script = _osascript_scripts(fake_run)[0]
    assert '\\"' in script  # quote escaped
    assert "\\\\" in script  # backslash escaped


def test_osascript_keystroke_script_shape_escapes_shell_injection(
    inserter, monkeypatch
):
    """The real _osascript_keystroke escaping (backslash first, then
    quote) keeps a hostile payload inside the AppleScript string
    literal, so it cannot break out and inject a command (issue #51
    lens SECURITY). Exercises the production code path, not a
    re-implementation of the escaping."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    hostile = '")do shell script "rm x"--'
    inserter.insert_text_typing(hostile)

    script = _osascript_scripts(fake_run)[0]
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
    # The script ends with the unescaped literal closing quote.
    assert script.endswith('"')


# --- osascript timeout constant (issue #58) ---
def test_run_osascript_passes_named_timeout_constant(inserter, monkeypatch):
    """_run_osascript passes _OSASCRIPT_TIMEOUT_S as the timeout kwarg to
    subprocess.run."""
    import kuiskaus.text_inserter as ti

    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)

    ti._run_osascript('keystroke "a"')

    # Floor sanity check: a System Events cold start was measured at
    # ~10s (issue #58), so the timeout must stay comfortably above it.
    assert ti._OSASCRIPT_TIMEOUT_S >= 10
    assert fake_run.call_args.kwargs["timeout"] == ti._OSASCRIPT_TIMEOUT_S


def test_run_osascript_timeout_message_derives_from_constant(inserter, monkeypatch):
    """TimeoutExpired message is built from the named constant (shows
    the constant's value, not a hardcoded string)."""
    import kuiskaus.text_inserter as ti

    fake_run = MagicMock(
        side_effect=subprocess.TimeoutExpired(
            cmd=["osascript"], timeout=ti._OSASCRIPT_TIMEOUT_S
        )
    )
    _patch_subprocess(monkeypatch, fake_run)

    ok, err = ti._run_osascript('keystroke "a"')

    assert ok is False
    assert "osascript timeout" in err
    assert str(ti._OSASCRIPT_TIMEOUT_S) in err


# --- failure paths: osascript fails, hint APPENDED (issue #58) ---
def test_osascript_failure_returns_false_with_error_and_tcc_hint(inserter, monkeypatch):
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


def test_osascript_1002_denial_records_error_and_tcc_hint(inserter, monkeypatch):
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
    message (derived from the constant) AND the TCC hint, with the
    underlying message first."""
    import kuiskaus.text_inserter as ti

    _patch_subprocess(
        monkeypatch,
        MagicMock(
            side_effect=subprocess.TimeoutExpired(
                cmd=["osascript"], timeout=ti._OSASCRIPT_TIMEOUT_S
            )
        ),
    )

    result = inserter.insert_text("hello world")

    assert result is False
    assert "osascript timeout" in inserter.last_error
    assert str(ti._OSASCRIPT_TIMEOUT_S) in inserter.last_error
    assert inserter.last_error.index("osascript timeout") < inserter.last_error.index(
        "Input Injection AND Accessibility"
    )


def test_tcc_revoked_hint_appended_on_injection_failure(inserter, monkeypatch):
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


def test_tahoe_hint_appended_when_axistrusted_true(inserter, monkeypatch):
    """osascript fails but AXIsProcessTrusted()==True: Tahoe hint is
    appended after the underlying error."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = True

    result = inserter.insert_text("hello world")

    assert result is False
    assert "denied" in inserter.last_error
    assert "Input Injection AND Accessibility" in inserter.last_error


def test_axistrusted_checked_once_per_session(inserter, monkeypatch):
    """Trust is cached after first failure; later failures reuse it."""
    _patch_subprocess(monkeypatch, _failing_osascript("denied"))
    app_services = sys.modules["ApplicationServices"]
    app_services.AXIsProcessTrusted.return_value = False

    inserter.insert_text("first failure")
    inserter.insert_text("second failure")

    assert app_services.AXIsProcessTrusted.call_count == 1


def test_last_error_cleared_at_start_of_each_insert_text(inserter, monkeypatch):
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
def test_insert_text_returns_false_on_pasteboard_failure(
    inserter, monkeypatch, pasteboard
):
    """NSPasteboard write False: no paste keystrokes (no osascript
    call), error recorded."""
    fake_run = _ok_osascript()
    _patch_subprocess(monkeypatch, fake_run)
    pasteboard.setString_forType_.return_value = False

    result = inserter.insert_text("hello world")

    assert result is False
    assert inserter.last_error is not None
    # The paste simulation (Cmd+V) must NOT run when the clipboard
    # write failed.
    assert fake_run.call_count == 0
