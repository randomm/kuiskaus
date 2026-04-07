"""Tests for apfel LLM post-processor."""

# Mock problematic imports before importing kuiskaus modules
import subprocess
import sys
from unittest.mock import MagicMock, patch

sys.modules["pyaudio"] = MagicMock()
sys.modules["mlx_whisper"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# ruff: noqa: E402 - Module import must come after mocks
from kuiskaus.postprocessor import (
    SYSTEM_PROMPT,
    _strip_meta_commentary,
    clean_with_apfel,
)


class TestStripMetaCommentary:
    def test_strips_sure_prefix(self):
        assert (
            _strip_meta_commentary("Sure, here you go.\nActual text.") == "Actual text."
        )

    def test_strips_here_prefix(self):
        assert (
            _strip_meta_commentary("Here is the result:\nActual text.")
            == "Actual text."
        )

    def test_strips_of_course_prefix(self):
        assert _strip_meta_commentary("Of course!\nclean text") == "clean text"

    def test_preserves_clean_text(self):
        assert _strip_meta_commentary("pytest runs fine.") == "pytest runs fine."

    def test_empty_string(self):
        assert _strip_meta_commentary("") == ""

    def test_multiline_clean(self):
        text = "Line one.\nLine two."
        assert _strip_meta_commentary(text) == text

    def test_leading_whitespace_meta_commentary_stripped(self):
        result = _strip_meta_commentary(" Sure, here is the result:\nActual text.")
        assert result == "Actual text."


class TestCleanWithApfel:
    def test_empty_text_returned_unchanged(self):
        assert clean_with_apfel("") == ""
        assert clean_with_apfel("   ") == "   "

    def test_successful_cleanup(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GitHub and pytest work great."
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = clean_with_apfel("get hub and pie test work great")
            assert result == "GitHub and pytest work great."
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "apfel"
            assert "-q" in call_args[0][0]
            assert "-s" in call_args[0][0]

    def test_nonzero_returncode_returns_original(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            original = "some text"
            assert clean_with_apfel(original) == original

    def test_timeout_returns_original(self):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("apfel", 10)
        ):
            original = "some text"
            assert clean_with_apfel(original) == original

    def test_file_not_found_returns_original(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            original = "some text"
            assert clean_with_apfel(original) == original

    def test_os_error_returns_original(self):
        with patch("subprocess.run", side_effect=OSError):
            original = "some text"
            assert clean_with_apfel(original) == original

    def test_unicode_decode_error_returns_original(self):
        with patch("subprocess.run", side_effect=ValueError("invalid utf-8")):
            original = "some text"
            assert clean_with_apfel(original) == original

    def test_empty_apfel_output_returns_original(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            original = "some text"
            assert clean_with_apfel(original) == original

    def test_meta_commentary_stripped_from_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Sure, here is the cleaned text:\nActual result."
        with patch("subprocess.run", return_value=mock_result):
            result = clean_with_apfel("raw text")
            assert result == "Actual result."

    def test_system_prompt_passed_to_apfel(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "cleaned"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            clean_with_apfel("test")
            call_args = mock_run.call_args[0][0]
            assert SYSTEM_PROMPT in call_args

    def test_xml_wrapping_in_input(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "cleaned"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            clean_with_apfel("hello world")
            call_args = mock_run.call_args[0][0]
            assert "<transcription>hello world</transcription>" in call_args

    def test_xml_special_chars_escaped_in_input(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "cleaned"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            clean_with_apfel("text with </transcription> tag")
            call_args = mock_run.call_args[0][0]
            # The raw closing tag should NOT appear unescaped
            assert (
                "</transcription>" not in call_args[-1]
                or "&lt;/transcription&gt;" in call_args[-1]
            )

    def test_custom_timeout(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "cleaned"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            clean_with_apfel("text", timeout=5)
            assert mock_run.call_args[1]["timeout"] == 5
