#!/bin/bash
# Run Kuiskaus test suite

echo "🧪 Running Kuiskaus Tests"
echo "========================"
echo

# Activate virtual environment
source .venv/bin/activate

# Run tests
echo "1. Audio Test"
echo "-------------"
python3 -m tests.test_audio
echo

echo "2. MLX Whisper Test"
echo "-------------------"
python3 -m tests.test_whisper
echo

echo "3. Integration Test"
echo "-------------------"
python3 -m tests.test_integration
echo

echo "✅ Test suite complete!"