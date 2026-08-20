#!/bin/bash
# Run Kuiskaus test suite (wraps pytest).
# Pass --hardware to also run the three manual hardware scripts, which
# require a microphone, downloaded models, and system permissions.

set -euo pipefail

echo "🧪 Running Kuiskaus Tests"
echo "========================"
echo

echo "Running pytest test suite"
uv run pytest tests/

if [[ "${1:-}" == "--hardware" ]]; then
    echo
    echo "Running manual hardware tests"
    echo "=============================="
    echo

    hardware_failures=()

    echo "1. Audio Test"
    echo "-------------"
    if ! uv run python -m tests.test_audio; then
        hardware_failures+=("test_audio")
    fi
    echo

    echo "2. MLX Whisper Test"
    echo "-------------------"
    if ! uv run python -m tests.test_whisper; then
        hardware_failures+=("test_whisper")
    fi
    echo

    echo "3. Integration Test"
    echo "-------------------"
    if ! uv run python -m tests.test_integration; then
        hardware_failures+=("test_integration")
    fi
    echo

    echo "Hardware test summary"
    echo "====================="
    for name in test_audio test_whisper test_integration; do
        if [[ " ${hardware_failures[*]:-} " == *" $name "* ]]; then
            echo "  FAILED: $name"
        else
            echo "  PASSED: $name"
        fi
    done
    echo

    if [[ ${#hardware_failures[@]} -gt 0 ]]; then
        echo "❌ Hardware tests failed: ${hardware_failures[*]}"
        exit 1
    fi
fi

echo "✅ Test suite complete!"
