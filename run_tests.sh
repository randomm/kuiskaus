#!/bin/bash
# Run Kuiskaus test suite

echo "🧪 Running Kuiskaus Tests"
echo "========================"
echo

# Activate virtual environment
source .venv/bin/activate

# Run unit tests (hardware-free)
python3 -m pytest tests/ -q
