#!/bin/bash
# Update dependencies using UV

source .venv/bin/activate
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
echo "✓ Dependencies updated"
