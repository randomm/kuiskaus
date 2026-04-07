# Kuiskaus 🎤

A fast, local speech-to-text application for Apple Silicon Macs using OpenAI's Whisper V3 Turbo model with MLX optimization. Hold a hotkey to record your voice, release to transcribe and automatically insert the text at your cursor position.

## Features

- **Blazing Fast**: Leverages Apple Silicon's Neural Engine via MLX for 8-15x real-time transcription
- **Global Hotkey**: Hold Control+Option (⌃⌥) to record from anywhere
- **Automatic Text Insertion**: Transcribed text is automatically typed at your cursor position
- **Menu Bar App**: Unobtrusive menu bar interface with easy access to settings
- **100% Local**: All processing happens on your Mac - no internet required after setup
- **Privacy First**: Your audio never leaves your device

## Requirements

- **macOS 12.0 or later**
- **Apple Silicon Mac (M1/M2/M3)** - Required
- **Python 3.8 or higher**
- **~1.5GB disk space** for the Whisper model
- **8GB RAM minimum** (16GB recommended)

⚠️ **Note**: This application is optimized exclusively for Apple Silicon and does not support Intel-based Macs.

## Installation

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

The setup script will:
- Verify you're running on Apple Silicon
- Install UV for ultra-fast package management
- Install system dependencies (portaudio, ffmpeg)
- Create a Python virtual environment
- Install all Python dependencies including MLX
- Download the Whisper V3 Turbo model (~1.5GB)
- Create launch scripts

3. Grant accessibility permissions when prompted:
   - Go to System Settings > Privacy & Security > Accessibility
   - Add and enable Terminal (or your terminal app)
   - Restart the app after granting permissions

### Manual Installation

If the setup script fails, you can install manually:

```bash
# Verify Apple Silicon
if [[ $(sysctl -n machdep.cpu.brand_string) != *"Apple"* ]]; then
    echo "Error: This app requires Apple Silicon"
    exit 1
fi

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install system dependencies
brew install portaudio ffmpeg

# Install Python packages
uv sync --group dev
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
- Change Whisper model size
- View usage statistics
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
5. **The text will be automatically inserted** at your cursor position

## Performance

With MLX optimization on Apple Silicon:
- **Whisper V3 Turbo**: 8-15x real-time (0.3-0.6s for 5s audio)
- **Model loads in ~1-2 seconds**
- **Leverages Neural Engine** for maximum efficiency
- **Minimal CPU usage** during transcription

## Configuration

### Available Models

The menu bar app allows you to switch between different Whisper models:
- **Turbo** (default): Fastest, optimized for speed
- **Base**: Smallest model, ultra-fast
- **Small**: Good balance of speed and accuracy
- **Medium**: Better accuracy, slower
- **Large**: Best accuracy, slowest

### Changing the Hotkey

To modify the hotkey, edit `kuiskaus/hotkey_listener_cgevent.py` and change the `required_modifiers`.

## Troubleshooting

### "Accessibility permissions required"
- Grant permissions in System Settings > Privacy & Security > Accessibility
- Add your terminal application to the list and enable it
- Restart the app after granting permissions

### No audio is being recorded
- Check that your microphone is working in other apps
- Ensure no other app is exclusively using the microphone
- Try selecting a different audio input device in System Settings

### Text not being inserted
- Some applications may block programmatic text input
- Try using the clipboard paste method (longer text is automatically pasted)
- Ensure the target application has focus when releasing the hotkey

### Model loading is slow
- First-time model download can take several minutes (~1.5GB)
- The model is cached locally after first download
- Subsequent loads take only 1-2 seconds

## Known Issues

- **Info.plist notification error**: If you see errors about Info.plist when running from a virtual environment, this is a known issue with rumps. The app will still work, but notifications may not display correctly.

## Privacy & Security

- **100% Local**: All speech processing happens on-device
- **No Internet Required**: Works completely offline after setup
- **No Data Collection**: Your audio and transcriptions never leave your Mac
- **Open Source**: Full source code available for inspection

## Development

### Project Structure
```
kuiskaus/
├── kuiskaus/               # Core application package
│   ├── audio_recorder.py   # PyAudio-based recording
│   ├── whisper_transcriber.py # MLX Whisper integration
│   ├── hotkey_listener_cgevent.py # Global hotkey detection
│   ├── text_inserter.py    # Text insertion at cursor
│   ├── app.py              # CLI application
│   └── menubar.py          # Menu bar application
├── tests/                  # Test suite
├── setup.sh                # Installation script
├── launch_kuiskaus.sh      # Menu bar launcher
├── launch_cli.sh           # CLI launcher
├── run_tests.sh            # Test runner
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependencies (auto-generated)
└── README.md
```

### Testing

Run the test suite:
```bash
./run_tests.sh
```

### Updating Dependencies

If you add or modify dependencies in `pyproject.toml`, regenerate the lockfile:
```bash
uv lock
uv sync --group dev
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for the Whisper model
- Apple for the MLX framework
- The Python community for excellent macOS integration libraries

---

**Note**: "Kuiskaus" is Finnish for "whisper" 🇫🇮