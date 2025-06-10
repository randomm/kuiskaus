#!/bin/bash
# Kuiskaus Setup Script - Apple Silicon Only

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

# Check for Apple Silicon
echo "Checking processor type..."
if [[ $(sysctl -n machdep.cpu.brand_string) != *"Apple"* ]]; then
    echo -e "${RED}Error: This application requires Apple Silicon (M1/M2/M3)${NC}"
    echo "Intel-based Macs are not supported."
    exit 1
fi
echo -e "${GREEN}✓ Apple Silicon detected${NC}"

# Install UV if not present
echo
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

# Pre-download Whisper model
echo
echo "Checking Whisper V3 Turbo model..."
python3 -c "
import os
import mlx_whisper

# Check if model is already cached
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
turbo_model = 'models--mlx-community--whisper-large-v3-turbo'

if os.path.exists(os.path.join(cache_dir, turbo_model)):
    print('✓ Turbo model already cached')
else:
    print('Downloading MLX Whisper V3 Turbo model...')
    print('Note: This will take 5-10 minutes (~1.5GB)')
    
    # Pre-download the model
    model_path = 'mlx-community/whisper-large-v3-turbo'
    model = mlx_whisper.load_models.load_model(model_path)
    print('✓ MLX Turbo model downloaded successfully')
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
python3 -m kuiskaus.menubar
EOL

chmod +x launch_kuiskaus.sh
echo -e "${GREEN}✓ Launch script created${NC}"

# Create command-line launcher
echo
echo "Creating CLI launch script..."
cat > launch_cli.sh << 'EOL'
#!/bin/bash
# Launch Kuiskaus CLI

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Launch the CLI version
python3 -m kuiskaus.app
EOL

chmod +x launch_cli.sh
echo -e "${GREEN}✓ CLI launch script created${NC}"

# Final instructions
echo
echo -e "${GREEN}✨ Setup complete!${NC}"
echo
echo "To use Kuiskaus:"
echo "1. Grant accessibility permissions when prompted"
echo "   (System Settings > Privacy & Security > Accessibility)"
echo "2. Launch the menu bar app with: ./launch_kuiskaus.sh"
echo "   Or use the CLI version with: ./launch_cli.sh"
echo
echo "Hold Control+Option (⌃⌥) to record, release to transcribe!"
echo
echo -e "${YELLOW}Note: This app requires Apple Silicon (M1/M2/M3)${NC}"