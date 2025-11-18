# Installation Guide

This project supports multiple encoding backends with automatic fallback. You don't need all of them - the system will automatically use the best available option.

## Quick Start

1. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install an encoding backend (try in this order):**
   
   **Option 1 (Recommended):** `ggwave-wheels` - Most reliable
   ```bash
   pip install ggwave-wheels
   ```
   
   **Option 2:** `ggwave-python` - Alternative ggwave package
   ```bash
   pip install ggwave-python
   ```
   
   **Option 3:** None - The code includes a simple FSK fallback that works without any additional packages (basic functionality)

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

### If ggwave packages won't install:
- The code will automatically fall back to a simple FSK encoder
- This works but is slower and less robust than ggwave
- For best results, try `ggwave-wheels` first as it's the most compatible

### If pyaudio won't install:
- Try `sounddevice` instead: `pip install sounddevice`
- On Linux, you may need: `sudo apt-get install portaudio19-dev`

### On Raspberry Pi:
- Make sure audio output is configured: `sudo raspi-config` → Advanced Options → Audio
- For best results, use a USB audio adapter if the built-in audio has issues

## Backend Priority

The system tries backends in this order:
1. `ggwave-wheels` (best compatibility)
2. `ggwave-python` (alternative)
4. Simple FSK (always works, basic functionality)

The first available backend will be used automatically.

