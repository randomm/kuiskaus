"""Shared KUISKAUS_DEBUG gating for per-event debug output (issue #22).

Both hotkey listeners import :data:`DEBUG` and :func:`debug` rather than
each defining their own copy of the env-var check and gated print.
"""

import os

# Any non-empty value of KUISKAUS_DEBUG enables per-event debug output.
# Read once at import; the listeners import the name into their own
# namespace (``from .debug import DEBUG``) so tests can flip the gating
# of a single listener with monkeypatch.setattr(module, "DEBUG", ...).
DEBUG = bool(os.environ.get("KUISKAUS_DEBUG"))


def debug(enabled: bool, *args: object) -> None:
    """Print *args* only when *enabled* is truthy.

    Callers pass their module-level ``DEBUG`` constant explicitly rather
    than this module's, so flipping one listener's binding does not leak
    into the other.
    """
    if enabled:
        print(*args)
