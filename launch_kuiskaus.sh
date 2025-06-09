#!/bin/bash
# Launch Kuiskaus

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Launch the application (menu bar version by default)
python3 -m kuiskaus.menubar
