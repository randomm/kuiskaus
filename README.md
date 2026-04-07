# Kuiskaus 🎤

Hold a hotkey to record your voice, release to transcribe and automatically paste the text at your cursor. All processing happens on your Apple Silicon Mac — no internet required after setup.

## Features

- **Fast transcription**: 8-15x real-time using Apple Silicon's Neural Engine
- **Global hotkey**: Hold Control+Option (⌃⌥) to record from anywhere
- **Automatic text insertion**: Transcribed text types at your cursor position
- **Menu bar interface**: Unobtrusive status control and settings
- **100% local**: Your audio never leaves your Mac

## Requirements

- macOS 12.0 or later
- Apple Silicon Mac (M1/M2/M3) — Intel not supported
- Python 3.8 or higher
- ~1.5GB disk space for the Whisper model
- 8GB RAM minimum (16GB recommended)

## Installation

### Quick Install

Run the setup script. It handles everything:

```bash
git clone https://github.com/randomm/kuiskaus.git
cd kuiskaus
./setup.sh
```

The script:
- Verifies you're on Apple Silicon
- Installs UV for fast package management
- Installs system dependencies (portaudio, ffmpeg)
- Installs Python dependencies including MLX
- Downloads the Whisper V3 Turbo model (~1.5GB)

After installation completes, grant accessibility permissions:

1. Open System Settings > Privacy & Security > Accessibility
2. Add your terminal app and enable it
3. Restart Kuiskaus

### Upgrading

Remove your virtual environment and reinstall:

```bash
rm -rf .venv && ./setup.sh
```

This ensures all dependencies update cleanly.

### Manual Installation

If the setup script fails, install manually:

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

Grant accessibility permissions as shown in Quick Install above.

## Usage

### Menu Bar App (Recommended)

```bash
./launch_kuiskaus.sh
```

Click the microphone icon (🎤) in your menu bar to:
- See current status
- Enable/disable speech recognition
- Change Whisper model size
- View usage statistics
- Quit the app

### CLI Version

```bash
./launch_cli.sh
```

### How It Works

1. Hold Control+Option (⌃⌥) to start recording
2. Speak clearly into your microphone
3. Release the keys to stop recording and transcribe
4. Text appears at your cursor position

## Performance

MLX optimization on Apple Silicon provides:
- **Whisper V3 Turbo**: 8-15x real-time (0.3-0.6s for 5s audio)
- **Model load**: ~1-2 seconds
- **Hardware acceleration**: Neural Engine utilization
- **Low CPU usage**: during transcription

## Configuration

### Model Selection

The menu bar app lets you switch between Whisper models:

| Model | Speed | Accuracy |
|-------|-------|----------|
| Turbo | Fastest | Good |
| Base | Fastest | Basic |
| Small | Fast | Balanced |
| Medium | Slower | Better |
| Large | Slowest | Best |

### Hotkey Modification

Edit `kuiskaus/hotkey_listener_cgevent.py` and change `required_modifiers`.

## Troubleshooting

### "Accessibility permissions required"

Grant permissions in System Settings > Privacy & Security > Accessibility. Add your terminal app and enable it. Restart Kuiskaus after granting permissions.

### No audio recorded

1. Verify your microphone works in other apps
2. Ensure no other app uses the microphone exclusively
3. Try a different audio input device in System Settings

### Text not inserted

1. Some applications block programmatic text input
2. Longer text uses clipboard paste automatically
3. Ensure the target application has focus when releasing the hotkey

### Slow model loading

First-time download takes several minutes (~1.5GB). The model caches locally after download. Subsequent loads take 1-2 seconds.

### Info.plist notification error

You may see Info.plist errors when running from a virtual environment. This is a known rumps issue. The app still works, but notifications may not display correctly.

## Privacy & Security

- **On-device processing**: All transcription happens locally
- **No internet required**: Works completely offline after setup
- **No data collection**: Your audio and transcriptions never leave your Mac
- **Open source**: Full source code available for inspection

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

### Setup

```bash
uv sync --group dev
```

This installs all dependencies including the `ty` type checker.

### Testing

Run the test suite:

```bash
./run_tests.sh
```

### Updating Dependencies

After modifying `pyproject.toml`, regenerate the lockfile:

```bash
uv lock
uv sync --group dev
```

### Code Quality

```bash
# Linting
uv run ruff check kuiskaus/ tests/

# Type checking
uv run ty check kuiskaus/

# Formatting
uv run ruff format kuiskaus/ tests/
```

## Contributing

Contributions are welcome! See the project structure above for code organization. Follow conventional commits format (`feat:`, `fix:`, `refactor:`, etc.) and name branches as `feature/issue-{number}-short-description`.

## License

MIT License — see LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Apple for the MLX framework
- The Python community for excellent macOS integration libraries

---

*"Kuiskaus" is Finnish for "whisper"* 🇫🇮