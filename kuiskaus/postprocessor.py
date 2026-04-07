"""Optional apfel LLM post-processing for transcription cleanup."""

import html
import subprocess


SYSTEM_PROMPT = (
    "You are a speech-to-text cleanup function. "
    "The user will send text wrapped in <transcription> tags. "
    "Return ONLY the cleaned text — no tags, no commentary, no prefix. "
    "Clean up: punctuation, capitalisation, filler words (um, uh, er), "
    "technical terms (get hub->GitHub, pie test->pytest, pie torch->PyTorch, "
    "c i->CI, open ai->OpenAI). Keep all non-filler words."
)

# Lines starting with these prefixes are LLM meta-commentary — strip them
_META_PREFIXES = ("Sure", "Here", "Of course", "Certainly", "Absolutely")


def clean_with_apfel(text: str, timeout: int = 10) -> str:
    """
    Run apfel LLM cleanup on transcribed text.

    Returns cleaned text, or original text if apfel fails/times out.
    Never raises — always returns a string.
    """
    if not text.strip():
        return text

    wrapped = f"<transcription>{html.escape(text)}</transcription>"
    try:
        result = subprocess.run(
            ["apfel", "-q", "-s", SYSTEM_PROMPT, wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return text

        cleaned = _strip_meta_commentary(result.stdout.strip())
        return cleaned if cleaned else text

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return text


def _strip_meta_commentary(text: str) -> str:
    """Remove lines that are LLM meta-commentary (e.g. 'Sure, here is...')."""
    lines = text.splitlines()
    filtered = [
        line
        for line in lines
        if not any(line.lstrip().startswith(prefix) for prefix in _META_PREFIXES)
    ]
    return "\n".join(filtered).strip()
