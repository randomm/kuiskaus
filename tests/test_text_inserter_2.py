"""test_text_inserter tests, part 2 of 2 (issue #70)."""

from unittest.mock import patch

from tests.support_text_inserter import (
    _set_strings_called,
)


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
