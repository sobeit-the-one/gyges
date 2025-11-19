# Installation Guide

This project supports multiple encoding backends with automatic fallback. You don't need all of them - the system will automatically use the best available option.

## Quick Start

1. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install GGWave support (optional, for GGWave encoding/decoding):**
   
   GGWave functionality requires the `gyges_encoder.exe` and `gyges_decoder.exe` binaries:
   - Build them from the C++ source (see `bin/README.md` for build instructions)
   - Place both binaries in one of these directories:
     - `bin/Release/`
     - `bin/Debug/`
     - `bin/MinSizeRel/`
   
   **Note:** If the binaries are not found, the code will automatically use a simple FSK fallback that works without any additional packages (basic functionality)

3. **Install an audio playback library (install one):**
   
   **Option 1 (Recommended):** `pyaudio`
   ```bash
   pip install pyaudio
   ```
   Note: On some systems, you may need to install system dependencies first:
   - Linux: `sudo apt-get install portaudio19-dev python3-pyaudio`
   - macOS: `brew install portaudio`
   - Windows: Usually works with pip directly
   
   **Option 2:** `sounddevice` - Alternative audio library
   ```bash
   pip install sounddevice
   ```

## Running the Application

Once dependencies are installed:

```bash
python app.py
```

Then open your browser to `http://localhost:5000` (or your Pi's IP address if running remotely).

## Troubleshooting

### If GGWave binaries are not available:
- The code will automatically fall back to a simple FSK encoder
- FSK works but is slower and less robust than GGWave
- To enable GGWave, build the binaries from source (see `bin/README.md`)

### If pyaudio won't install:
- Try `sounddevice` instead: `pip install sounddevice`
- On Linux, you may need: `sudo apt-get install portaudio19-dev`

### On Raspberry Pi:
- Make sure audio output is configured: `sudo raspi-config` → Advanced Options → Audio
- For best results, use a USB audio adapter if the built-in audio has issues

## Backend Priority

The system tries backends in this order:
1. **GGWave** (via `gyges_encoder.exe` and `gyges_decoder.exe` binaries) - Best quality, 140 bytes max
2. **Simple FSK** (always works, basic functionality) - Up to ~6.5KB, configurable

The first available backend will be used automatically. If GGWave binaries are not found, FSK will be used as a fallback.

