# 🎵 Gyges - Audio Data Transmitter

A Python web application that encodes data into audio waveforms and decodes audio back into data using data-over-sound technology.

## Features

- **Multiple Encoders**: 
  - **GGWave** (up to 140 bytes, high quality) - Uses `gyges_encoder.exe` and `gyges_decoder.exe` binaries
    - 12 protocol options: Audible (Normal/Fast/Fastest), Ultrasound (silent), Dual Tone, Multi Tone
  - **FSK** (up to ~6.5KB, always available) - Pure Python implementation
    - Configurable bit duration, frequencies, and sample rate
- **Encoding & Decoding**: 
  - Encode data to WAV files
  - Decode WAV files back to data (both GGWave and FSK supported)
- **Web Interface**: Easy-to-use browser-based UI
- **Persistent History**: SQLite database stores all transmissions with metadata
- **File Support**: Handle text and binary files (images, documents, etc.)
- **WAV Export/Import**: Download generated waveforms or upload recorded audio

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install audio playback (recommended):**
   ```bash
   pip install pyaudio
   ```

3. **Ensure `gyges_encoder.exe` and `gyges_decoder.exe` are available (optional, for GGWave support):**
   - Place both binaries in `bin/Release/`, `bin/Debug/`, or `bin/MinSizeRel/`
   - If not found, only FSK encoding/decoding will be available

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   ```
   http://localhost:5000
   ```

## Usage

### Encoding
1. Select an encoder (Auto, GGWave, or FSK)
2. Adjust parameters:
   - **GGWave**: Choose protocol (Audible/Ultrasound/DT/MT) and volume
   - **FSK**: Set bit duration, frequency 0, frequency 1
3. Enter text or upload a file
4. Click "Generate from Text/File"
5. Play the audio or download the WAV file

### Decoding
1. Upload a WAV file (recorded or downloaded)
2. Select decoder (Auto-detect, GGWave, or FSK)
3. For FSK: Match the encoding parameters (bit duration, frequencies)
4. For GGWave: Uses 48kHz sample rate by default
5. Optionally specify output filename
6. Click "Decode Audio"
7. Download the decoded data

## Limitations

- **GGWave**: 
  - 140 bytes maximum
  - Full encoding and decoding support via `gyges_encoder.exe` and `gyges_decoder.exe`
  - Requires both binaries in `bin/Release/`, `bin/Debug/`, or `bin/MinSizeRel/`
- **FSK**: 
  - ~6.5KB at default settings (10ms bit duration)
  - Dynamic limit based on bit duration (keeps WAV files under 50MB)
  - Lower bit duration = more data fits, but faster transmission
- **Audio Quality**: Clean recordings work best for decoding

## Documentation

See [INSTALL.md](INSTALL.md) for detailed installation instructions and troubleshooting.

## Use Cases

- Transmit data over phone lines using audio
- Air-gapped data transfer via speakers/microphones
- Embed data in audio files
- Educational experiments with data-over-sound

## License

MIT License - See LICENSE file for details

