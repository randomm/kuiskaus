#!/bin/bash
# Run Kuiskaus tests

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

echo "🧪 Running Kuiskaus Tests"
echo "========================"
echo

# Run each test
echo "1. Audio Recording Test"
echo "-----------------------"
python3 -m tests.test_audio
echo

echo "2. Whisper Model Test"
echo "---------------------"
python3 -m tests.test_whisper
echo

echo "3. System Integration Test"
echo "--------------------------"
python3 -m tests.test_integration