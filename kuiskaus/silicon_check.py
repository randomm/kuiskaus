"""Shared Apple Silicon check."""

import subprocess


def check_apple_silicon() -> bool:
    """Check if running on Apple Silicon"""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Apple" in result.stdout
    except (subprocess.SubprocessError, OSError):
        return False
