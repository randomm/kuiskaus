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

    echo "1. Audio Test"
    echo "-------------"
    uv run python -m tests.test_audio
    echo

    echo "2. MLX Whisper Test"
    echo "-------------------"
    uv run python -m tests.test_whisper
    echo

    echo "3. Integration Test"
    echo "-------------------"
    uv run python -m tests.test_integration
    echo
fi

echo "✅ Test suite complete!"
