#!/bin/bash
# Kuiskaus Setup Script

set -e

echo "🎤 Kuiskaus Setup Script"
echo "========================"
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This application only runs on macOS${NC}"
    exit 1
fi

# Install UV if not present
echo "Checking for UV (fast Python package installer)..."
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV not found. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Failed to install UV${NC}"
        echo "Please install UV manually from https://github.com/astral-sh/uv"
        exit 1
    fi
    
    echo -e "${GREEN}✓ UV installed successfully${NC}"
else
    echo -e "${GREEN}✓ UV already installed ($(uv --version))${NC}"
fi

# Check for Homebrew
echo
echo "Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "${GREEN}✓ Homebrew installed${NC}"
fi

# Install system dependencies
echo
echo "Installing system dependencies..."
brew install portaudio ffmpeg

# Create virtual environment with UV
echo
echo "Creating Python virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
    rm -rf .venv
fi

uv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Install Python dependencies with UV
echo
echo "Installing Python dependencies (this will be fast!)..."
source .venv/bin/activate

# Generate requirements.txt from requirements.in
if [ -f "requirements.in" ]; then
    echo "Compiling requirements..."
    uv pip compile requirements.in -o requirements.txt
fi

# Install dependencies
echo "Installing dependencies..."
uv pip sync requirements.txt

# Try to install MLX-optimized whisper for Apple Silicon
echo
echo "Checking processor type..."
if [[ $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    echo "Apple Silicon detected. Installing MLX-optimized Whisper..."
    if uv pip install mlx-whisper; then
        echo -e "${GREEN}✓ MLX-optimized Whisper installed (Apple Silicon acceleration enabled)${NC}"
    else
        echo -e "${YELLOW}MLX-optimized Whisper not available (using standard Whisper)${NC}"
    fi
else
    echo "Intel Mac detected. Using standard Whisper implementation."
fi

# Pre-download Whisper model
echo
echo "Pre-downloading Whisper V3 Turbo model..."
python3 -c "
import sys

# Try MLX first (best for Apple Silicon)
try:
    import mlx_whisper
    print('Downloading MLX Whisper V3 Turbo model...')
    print('Note: First download may take 5-10 minutes depending on your connection')
    
    # Pre-download the model
    model_path = 'mlx-community/whisper-large-v3-turbo'
    model = mlx_whisper.load_models.load_model(model_path)
    print('✓ MLX Turbo model downloaded successfully')
    sys.exit(0)
except Exception as e:
    print(f'MLX model download failed: {e}')

# Fallback to faster-whisper
try:
    from faster_whisper import WhisperModel
    print('Downloading faster-whisper V3 Turbo model...')
    model = WhisperModel('large-v3-turbo', device='auto', compute_type='int8')
    print('✓ Faster-whisper Turbo model downloaded successfully')
except Exception as e:
    print(f'Model download failed: {e}')
    print('The model will be downloaded on first use.')
"

# Create launch script
echo
echo "Creating launch script..."
cat > launch_kuiskaus.sh << 'EOL'
#!/bin/bash
# Launch Kuiskaus

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Launch the application (menu bar version by default)
python3 kuiskaus_menubar.py
EOL

chmod +x launch_kuiskaus.sh
echo -e "${GREEN}✓ Launch script created${NC}"

# Create command-line launcher
echo
echo "Creating command-line launcher..."
cat > launch_cli.sh << 'EOL'
#!/bin/bash
# Launch Kuiskaus CLI version

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Launch the CLI version
python3 kuiskaus_app.py "$@"
EOL

chmod +x launch_cli.sh
echo -e "${GREEN}✓ CLI launcher created${NC}"

# Create dependency update script
cat > update_deps.sh << 'EOL'
#!/bin/bash
# Update dependencies using UV

source .venv/bin/activate
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
echo "✓ Dependencies updated"
EOL

chmod +x update_deps.sh

# Check accessibility permissions
echo
echo -e "${YELLOW}⚠️  Important: Accessibility Permissions${NC}"
echo "Kuiskaus requires accessibility permissions to:"
echo "  • Listen for global hotkeys"
echo "  • Insert text at cursor position"
echo
echo "You will be prompted to grant these permissions when you first run the app."
echo "Go to: System Preferences > Security & Privacy > Privacy > Accessibility"
echo

# Success message
echo -e "${GREEN}✅ Setup complete!${NC}"
echo
echo "⚡ UV provides 10-100x faster package installation!"
echo
echo "To run Kuiskaus:"
echo "  • Menu bar app: ./launch_kuiskaus.sh"
echo "  • CLI version:  ./launch_cli.sh"
echo
echo "To run tests:"
echo "  • ./test_audio_recording.py"
echo "  • ./test_whisper_model.py"
echo "  • ./test_system_integration.py"
echo
echo "Hotkey: Hold Control+Option (⌃⌥) to record, release to transcribe"
echo
echo "Enjoy using Kuiskaus! 🎤"