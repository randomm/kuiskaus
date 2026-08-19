"""Shared pytest infrastructure for hardware-free unit tests.

Pre-mocks macOS hardware dependencies (pyaudio, the MLX/parakeet model
modules, and the AppKit/Quartz/pyobjc layers) *before* any test module
imports a ``kuiskaus`` module.  Each test file currently does its own
``sys.modules`` mocking; this conftest centralizes that so new unit
tests (e.g. hotkey ordering tests) can import the package modules
without a microphone, GPU models, or a macOS event loop.
"""

import sys
from unittest.mock import MagicMock

import numpy as _np
import pytest

# `numpy` is mocked out at module level in test_parakeet.py /
# test_voxtral.py; if such a module is imported before its counterpart's
# module-level ``import numpy`` runs, that import would bind the mock
# instead of the real library (breaking ``np.float32`` etc.).  Cache the
# real reference here (conftest always imports first) and restore it via
# the autouse fixture below so every module's ``import numpy`` sees real
# numpy regardless of collection order.
REAL_NUMPY = _np

# --- Hardware / model layer: mock before any kuiskaus import ---------------
_MOCK_MODULES = {
    "pyaudio": MagicMock(),
    "mlx_whisper": MagicMock(),
    "mlx_whisper.load_models": MagicMock(),
    "mlx_voxtral": MagicMock(),
    "parakeet_mlx": MagicMock(),
    "parakeet_mlx.audio": MagicMock(),
    # macOS event / app layer (hotkey and text-insertion tests).  Mock the
    # whole pyobjc stack so `import rumps` (which drags in Foundation and
    # objc) works hardware-free; AppKit's NSApplication must not be a
    # MagicMock or `isinstance(app, NSApplication)` checks in rumps raise.
    "PyObjCTools": MagicMock(),
    "PyObjCTools.AppHelper": MagicMock(),
    "PyObjCTools.KeyValueCoding": MagicMock(),
    "objc": MagicMock(),
    "Foundation": MagicMock(),
    "ApplicationServices": MagicMock(),
    "Quartz": MagicMock(),
    "AppKit": MagicMock(),
}
_MOCK_MODULES["parakeet_mlx.audio"].get_logmel = MagicMock(return_value=MagicMock())
_MOCK_MODULES["parakeet_mlx"].from_pretrained = MagicMock(return_value=MagicMock())
_MOCK_MODULES["AppKit"].NSApplication = type(
    "NSApplication", (object,), {}
)

sys.modules.update(_MOCK_MODULES)


@pytest.fixture(autouse=True)
def _restore_real_numpy():
    """Put real numpy back after a test module mocked it out."""
    yield
    sys.modules["numpy"] = REAL_NUMPY
