#!/bin/bash
# Launch Kuiskaus CLI

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Launch the CLI version
echo "Starting Kuiskaus CLI..."
python3 -u -m kuiskaus.app "$@"
