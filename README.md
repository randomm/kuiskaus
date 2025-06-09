# Kuiskaus 🎤

A fast, local speech-to-text application for macOS using OpenAI's Whisper V3 Turbo model. Hold a hotkey to record your voice, release to transcribe and automatically insert the text at your cursor position.

## Features

- **Fast Transcription**: Uses Whisper V3 Turbo model optimized for Apple Silicon
- **Global Hotkey**: Hold Control+Option (⌃⌥) to record from anywhere
- **Automatic Text Insertion**: Transcribed text is automatically typed at your cursor position
- **Menu Bar App**: Unobtrusive menu bar interface with easy access to settings
- **Privacy First**: All processing happens locally on your Mac - no internet required
- **MLX Optimization**: Leverages Apple Silicon neural engines for maximum performance

## Requirements

### System Requirements
- macOS 10.15 or later
- Python 3.8 or higher
- **Processor Support**:
  - ✅ Apple Silicon (M1/M2/M3) - Best performance with MLX optimization
  - ✅ Intel Macs - Fully supported (without MLX optimization)
- ~1.5GB disk space for the Whisper model
- At least 8GB RAM (16GB recommended)

### Dependencies (automatically installed by setup script)
- **Homebrew** - Package manager for macOS (will be installed if not present)
- **PortAudio** - Required for audio recording (`brew install portaudio`)
- **FFmpeg** - Required for audio processing (`brew install ffmpeg`)
- **Python packages**:
  - `pyobjc-core` and related frameworks - For macOS system integration
  - `pyaudio` - Audio recording interface
  - `numpy` - Numerical processing
  - `openai-whisper` - Whisper speech recognition model
  - `mlx-whisper` (optional) - MLX-optimized version for Apple Silicon
  - `rumps` - Menu bar application framework

## Installation

### Prerequisites
Ensure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

### Quick Install

1. Clone this repository:
```bash
git clone https://github.com/randomm/kuiskaus.git
cd kuiskaus
```

2. Run the setup script:
```bash
./setup.sh
```

The setup script will automatically:
- Install [UV](https://github.com/astral-sh/uv) for fast package management
- Check system compatibility (Intel or Apple Silicon)
- Install Homebrew if not present
- Install system dependencies (portaudio, ffmpeg)
- Create a Python virtual environment
- Install all Python dependencies (10-100x faster with UV!)
- Download the Whisper V3 Turbo model (~1.5GB)
- Create launch scripts

3. Grant accessibility permissions when prompted:
   - Go to System Preferences > Security & Privacy > Privacy > Accessibility
   - Add and enable the Terminal app (or your terminal of choice)
   - You may need to restart the app after granting permissions

### Manual Installation (if setup script fails)

```bash
# Install UV (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install portaudio ffmpeg

# Create virtual environment with UV
uv venv
source .venv/bin/activate

# Install Python packages with UV
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt

# For Apple Silicon only (optional)
uv pip install mlx-whisper
```

## Usage

### Menu Bar App (Recommended)

Launch the menu bar version:
```bash
./launch_kuiskaus.sh
```

The app will appear in your menu bar as a microphone icon (🎤). Click it to:
- See current status
- Enable/disable speech recognition
- Change Whisper model
- View statistics
- Quit the app

### CLI Version

For a command-line interface:
```bash
./launch_cli.sh
```

### How to Use

1. **Start the app** using one of the methods above
2. **Hold Control+Option (⌃⌥)** to start recording
3. **Speak clearly** into your microphone
4. **Release the keys** to stop recording and transcribe
5. **The transcribed text will be automatically inserted** at your cursor position

## Configuration

### Changing the Hotkey

To modify the hotkey combination, edit `hotkey_listener.py` and change the `required_modifiers` in the `__init__` method.

### Using Different Models

The menu bar app allows you to switch between different Whisper models:
- **Turbo** (default): Fastest, uses V3 Turbo
- **Base**: Smallest model, very fast
- **Small**: Good balance of speed and accuracy
- **Medium**: Better accuracy, slower
- **Large**: Best accuracy, slowest

## Performance

### Apple Silicon Macs (with MLX optimization)
- Whisper V3 Turbo typically transcribes at 8-15x real-time
- A 5-second recording transcribes in ~0.3-0.6 seconds
- Leverages Neural Engine for maximum efficiency
- Model loads in ~1-2 seconds on first use

### Intel Macs
- Whisper V3 Turbo typically transcribes at 2-5x real-time
- A 5-second recording transcribes in ~1-2.5 seconds
- Uses CPU-based processing (GPU acceleration if available)
- Model loads in ~3-5 seconds on first use

**Note**: Performance varies based on audio quality, background noise, and system load

## Testing

The project includes diagnostic tests to help troubleshoot issues:

```bash
# Run all tests
./run_tests.sh

# Or run individual tests
python3 -m tests.test_audio       # Test audio recording
python3 -m tests.test_whisper     # Test Whisper model
python3 -m tests.test_integration # Test system integration
```

Run these tests after installation to verify everything is working correctly.

## Known Issues

- **Info.plist notification error**: If you see errors about Info.plist when running from a virtual environment, this is a known issue with rumps. The app will still work, but notifications may not display correctly.
- **First-time model loading**: The first transcription after starting the app may take a few seconds as the model loads into memory.

## Troubleshooting

### "Accessibility permissions required"
- Grant permissions in System Preferences > Security & Privacy > Accessibility
- Add your terminal application to the list and enable it
- Restart the app after granting permissions

### No audio is being recorded
- Check that your microphone is working in other apps
- Ensure no other app is exclusively using the microphone
- Try selecting a different audio input device in System Preferences

### Text not being inserted
- Some applications may block programmatic text input
- Try using the clipboard paste method (longer text is automatically pasted)
- Ensure the target application has focus when releasing the hotkey

### Model loading is slow
- First-time model download can take several minutes (~1.5GB)
- The model is cached locally after first download
- MLX optimization is only available on Apple Silicon Macs

### PortAudio installation issues
If you encounter errors installing PyAudio:
```bash
# On Intel Macs
brew install portaudio
pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio

# On Apple Silicon Macs
brew install portaudio
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```

## Privacy & Security

- **100% Local**: All speech processing happens on your device
- **No Internet Required**: Works completely offline after setup
- **No Data Collection**: Your audio and transcriptions never leave your Mac
- **Open Source**: Full source code available for inspection

## Development

### Project Structure
```
kuiskaus/
├── kuiskaus/               # Core application package
│   ├── __init__.py
│   ├── audio_recorder.py   # Audio recording functionality
│   ├── whisper_transcriber.py # Whisper model integration
│   ├── hotkey_listener.py  # Global hotkey detection
│   ├── text_inserter.py    # Text insertion at cursor
│   ├── app.py              # CLI application
│   └── menubar.py          # Menu bar application
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_audio.py       # Audio system tests
│   ├── test_whisper.py     # Whisper model tests
│   └── test_integration.py # System integration tests
├── setup.sh                # Installation script
├── launch_kuiskaus.sh      # Menu bar launcher
├── launch_cli.sh           # CLI launcher
├── run_tests.sh            # Test runner
├── requirements.in         # Direct dependencies
├── requirements.txt        # Locked dependencies
├── LICENSE
└── README.md
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for the amazing Whisper model
- Apple for MLX framework
- The Python community for excellent macOS integration libraries

---

**Note**: "Kuiskaus" is Finnish for "whisper" 🇫🇮