# 🎵 Gyges - Audio Data Transmitter

A Python web application that encodes data into audio waveforms and decodes audio back into data using data-over-sound technology.

## Features

- **Multiple Encoders**: 
  - GGWave (up to 140 bytes, high quality)
  - FSK (up to ~6.5KB, always available fallback)
- **Bidirectional**: Encode data to WAV files and decode WAV files back to data
- **Web Interface**: Easy-to-use browser-based UI
- **Persistent History**: SQLite database stores all transmissions
- **File Support**: Handle text and binary files (images, documents, etc.)
- **WAV Export/Import**: Download generated waveforms or upload recorded audio

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install an encoder (recommended):**
   ```bash
   pip install ggwave-wheels
   ```

3. **Install audio playback (recommended):**
   ```bash
   pip install pyaudio
   ```

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
2. Adjust parameters if needed
3. Enter text or upload a file
4. Click "Generate from Text/File"
5. Play the audio or download the WAV file

### Decoding
1. Upload a WAV file (recorded or downloaded)
2. Select decoder (Auto-detect recommended)
3. Optionally specify output filename
4. Click "Decode Audio"
5. Download the decoded data

## Limitations

- **GGWave**: 140 bytes maximum
- **FSK**: ~6.5KB at default settings (adjust bit duration for more/less)
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

