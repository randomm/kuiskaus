"""Unit tests for the shared Apple Silicon check (issue #22).

Verifies the extracted implementation behaves the same way the entry
points relied on: true when `sysctl` reports an Apple brand string,
false otherwise, and false (not an exception) when sysctl is missing.
"""

import subprocess
from unittest.mock import patch

from kuiskaus.silicon_check import check_apple_silicon


def test_true_on_apple_brand_string():
    with patch(
        "kuiskaus.silicon_check.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Apple M3\n"
        ),
    ) as mock_run:
        assert check_apple_silicon() is True

    mock_run.assert_called_once_with(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_false_on_intel_brand_string():
    with patch(
        "kuiskaus.silicon_check.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Intel(R) Core(TM) i9 CPU\n"
        ),
    ):
        assert check_apple_silicon() is False


def test_false_when_sysctl_raises():
    with patch(
        "kuiskaus.silicon_check.subprocess.run",
        side_effect=FileNotFoundError("sysctl not found"),
    ):
        assert check_apple_silicon() is False
