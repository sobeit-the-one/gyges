# Gyges C++ - Data-to-Audio Conversion

A CMake C++ project that converts file content into audio using multiple modulation methods including GGWave and Frequency Shift Keying (FSK). Includes both encoders and decoders with round-trip validation.

## Features

- **GGWave Encoding/Decoding**: Uses the ggwave library for robust data-over-sound transmission with error correction
- **FSK Encoding/Decoding**: Custom implementation of Binary Frequency Shift Keying (FSK) modulation (matches Python implementation)
- **WAV Output**: Generates standard WAV audio files
- **Command-line Interface**: Uses argparse for flexible command-line argument parsing
- **Testing**: Comprehensive test suite using doctest with round-trip encoding/decoding validation

## Requirements

- CMake 3.10 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- Git (for submodules)

## Building

### Initial Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd gyges_cpp
```

2. Initialize and update submodules:

```bash
git submodule update --init --recursive
```

### Build Instructions

1. Create a build directory:

```bash
mkdir build
cd build
```

2. Configure with CMake:

```bash
cmake ..
```

3. Build the project:

```bash
cmake --build .
```

On Windows with Visual Studio:

```bash
cmake --build . --config Release
```

On Unix-like systems:

```bash
make
```

## Usage

The project provides two separate executables:

- **gyges_encoder**: Converts files to audio (encoding)
- **gyges_decoder**: Converts audio back to files (decoding)

### Encoding (File → Audio)

Convert a file to audio using GGWave (default):

```bash
./gyges_encoder input.txt -o output.wav
```

Convert using FSK modulation:

```bash
./gyges_encoder input.txt -m fsk -o output.wav
```

Generate both GGWave and FSK outputs:

```bash
./gyges_encoder input.txt -m both
```

### Decoding (Audio → File)

Decode audio file back to data:

```bash
./gyges_decoder input.wav -m fsk -o output.bin
```

Decode GGWave audio:

```bash
./gyges_decoder input.wav -m ggwave -o output.bin
```

### JSON Output Mode

For programmatic use (e.g., from Python), use the `--json` flag to get structured JSON output:

```bash
./gyges_encoder input.txt -o output.wav --json
```

**Success response:**
```json
{"success":"true","method":"fsk","output_file":"output.wav","samples":"44100","sample_rate":"44100"}
```

**Error response:**
```json
{"success":"false","error_code":"2","error":"Cannot open input file: input.txt"}
```

When using `--json`, all human-readable messages go to stderr, and JSON goes to stdout for easy parsing.

### Command-line Options

#### gyges_encoder

```text
positional arguments:
  input                 Input file to convert to audio

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output WAV file (default: input filename with .wav extension)
  -m, --method METHOD    Encoding method: 'ggwave', 'fsk', or 'both' (default: ggwave)
  --json                 Output JSON format instead of human-readable messages
  
FSK Parameters:
  --f0 F0               FSK frequency for bit '0' in Hz (default: 1200)
  --f1 F1               FSK frequency for bit '1' in Hz (default: 1800)
  --bit-duration DUR    FSK bit duration in seconds (default: 0.01 = 10ms = 100 baud)
  
GGWave Parameters:
  --protocol PROTOCOL   GGWave protocol ID (default: 1)
  --volume VOLUME       GGWave volume [1-100] (default: 50)
  
Common Parameters:
  --sample-rate RATE    Audio sample rate in Hz (default: 44100)
```

#### gyges_decoder

```text
positional arguments:
  input                 Input WAV audio file to decode

required arguments:
  -o, --output OUTPUT   Output file path (required)
  -m, --method METHOD   Decoding method: 'ggwave' or 'fsk' (required)

optional arguments:
  -h, --help            Show help message
  --json                 Output JSON format instead of human-readable messages
  
FSK Parameters (must match encoder):
  --f0 F0               FSK frequency for bit '0' in Hz (default: 1200)
  --f1 F1               FSK frequency for bit '1' in Hz (default: 1800)
  --bit-duration DUR    FSK bit duration in seconds (default: 0.01)
  --sample-rate RATE    Audio sample rate in Hz (default: 44100)
  --no-preamble          Disable preamble detection
  --preamble-bits N      Number of preamble bits to skip (default: 16)
  
GGWave Parameters:
  --ggwave-sample-rate RATE  GGWave sample rate in Hz (default: 48000)
```

### Error Codes

Both executables return the following exit codes:

- **0**: Success
- **1**: Invalid command-line arguments
- **2**: File I/O error (cannot open/read/write file)
- **3**: Encoding/decoding failure (data processing error)
- **4**: Unsupported method or format
- **5**: Configuration error (invalid parameters)

When using `--json`, the error code is also included in the JSON response.

### Examples

**Encode with FSK using custom frequencies:**

```bash
./gyges_encoder data.bin -m fsk --f0 1000 --f1 2000 --bit-duration 0.005 -o output.wav
```

**Encode with GGWave using custom protocol and volume:**

```bash
./gyges_encoder message.txt --protocol 2 --volume 75 -o message.wav
```

**Encode with high sample rate FSK:**

```bash
./gyges_encoder file.bin -m fsk --sample-rate 96000 -o high_quality.wav
```

**Decode FSK audio back to file:**

```bash
./gyges_decoder output.wav -m fsk --f0 1000 --f1 2000 --bit-duration 0.005 -o decoded.bin
```

**Encode and decode with JSON output (for Python integration):**

```bash
# Encode
./gyges_encoder input.txt -o output.wav --json

# Decode
./gyges_decoder output.wav -m fsk -o decoded.txt --json
```

## FSK Modulation Details

Frequency Shift Keying (FSK) is a frequency modulation scheme where digital information is encoded by shifting the carrier frequency between discrete frequencies:

- **Bit '0'**: Transmitted at frequency `f0` (default: 1200 Hz)
- **Bit '1'**: Transmitted at frequency `f1` (default: 1800 Hz)

The encoder implementation matches the Python reference:

1. Converts input data to a binary bit stream (MSB first)
2. Adds an optional synchronization preamble (alternating 0/1 pattern)
3. For each bit, generates a sine wave at the appropriate frequency
4. Each bit period lasts `bit_duration` seconds (default: 0.01s = 10ms = 100 baud)
5. Generates `sample_rate * bit_duration` samples per bit
6. Normalizes the output to range [-1.0, 1.0]

### FSK Parameters

- **f0, f1**: Carrier frequencies in Hz. Should be within the audio range (typically 300-3400 Hz for telephone quality, up to 20 kHz for high-quality audio)
- **Bit Duration**: Duration of each bit in seconds. Default is 0.01s (10ms), which equals 100 baud (bits per second). Lower values = faster transmission but less reliable. Higher values = slower but more reliable.
- **Sample Rate**: Audio sample rate. Standard rates: 44100 Hz (CD quality), 48000 Hz (professional), 96000 Hz (high quality)
- **Preamble**: Optional synchronization pattern (16 alternating bits by default) to help with decoding

### FSK Decoder

The FSK decoder uses correlation-based frequency detection:
- Analyzes each bit period to determine which frequency (f0 or f1) is present
- Skips preamble bits if present
- Converts detected bits back to bytes (MSB first)

## GGWave Protocols

GGWave supports multiple protocols optimized for different use cases:

- Protocol 1 (default): DT_NORMAL - Balanced speed and reliability
- Protocol 2: DT_FAST - Faster transmission
- Protocol 3: DT_FASTEST - Fastest transmission
- And more...

See the ggwave documentation for full protocol details.

## Running Tests

Build and run the test suite:

```bash
cd build
cmake --build . --target gyges_cpp_tests
./gyges_cpp_tests
```

Or use CTest:

```bash
ctest
```

The test suite includes:
- Unit tests for encoders, decoders, WAV writer, and WAV reader
- Round-trip tests that validate data can be encoded and decoded successfully
- FSK encoding/decoding validation
- GGWave encoding/decoding validation
- JSON output format validation

## Project Structure

```text
gyges_cpp/
├── CMakeLists.txt          # Root CMake configuration
├── src/                    # Source files
│   ├── encoder_main.cpp   # Encoder application
│   ├── decoder_main.cpp   # Decoder application
│   ├── encoder.h/cpp      # Encoder interface and GGWave implementation
│   ├── decoder.h/cpp      # Decoder interface and GGWave implementation
│   ├── fsk_encoder.h/cpp  # FSK encoder implementation
│   ├── fsk_decoder.h/cpp  # FSK decoder implementation
│   ├── wav_writer.h/cpp   # WAV file writer
│   ├── wav_reader.h/cpp   # WAV file reader
│   └── json_output.h/cpp  # JSON output utility
├── tests/                 # Test files
│   ├── test_main.cpp      # Main test file
│   ├── test_encoder.cpp  # Encoder tests
│   ├── test_fsk.cpp       # FSK encoder tests
│   ├── test_wav.cpp      # WAV writer tests
│   └── test_roundtrip.cpp # Round-trip encoding/decoding tests
└── external/             # Git submodules
    ├── ggwave/           # GGWave library
    ├── argparse/         # Argument parser
    └── doctest/          # Testing framework
```

## Dependencies

All dependencies are managed as Git submodules:

- **ggwave**: Data-over-sound library with multi-frequency FSK support
  - Repository: https://github.com/ggerganov/ggwave
- **argparse**: Single-header C++ argument parser
  - Repository: https://github.com/p-ranav/argparse
- **doctest**: Lightweight C++ testing framework
  - Repository: https://github.com/doctest/doctest

## Comparison with Other Libraries

### GGWave

- **Library**: C++ library with multi-frequency FSK modulation
- **Use Case**: Short-range data transmission, device pairing
- **Bitrate**: Moderate (protocol-dependent)
- **Error Correction**: Built-in Reed-Solomon error correction
- **Status**: ✅ Implemented in this project (encoder and decoder)

### FSK (Simple Binary FSK)

- **Implementation**: Custom C++ implementation (matches Python reference)
- **Use Case**: Simple, reliable data transmission
- **Bitrate**: Low to moderate (100-1200 baud typical, configurable via bit duration)
- **Error Correction**: None (relies on preamble for sync)
- **Status**: ✅ Implemented in this project (encoder and decoder)

### amodem (Python)

- **Library**: Python library using OFDM (Orthogonal Frequency-Division Multiplexing)
- **Use Case**: High-speed air-gapped communication
- **Modulation**: OFDM with BPSK, 4-PSK, 16-QAM, 64-QAM, 256-QAM
- **Bitrate**: Up to 80 kbps (much higher than FSK)
- **Error Correction**: CRC-32 checksums per 250-byte frame
- **Sampling Rates**: 8/16/32 kHz
- **Status**: ⚠️ Python-only, not integrated (reference implementation)
- **Repository**: https://github.com/romanz/amodem

**Note**: amodem uses OFDM, which is significantly more complex than simple FSK. It requires:

- FFT/IFFT for OFDM symbol generation
- Multiple subcarriers
- Complex modulation schemes (QAM)
- Frame synchronization
- Channel estimation and equalization

Implementing full OFDM support in C++ would be a substantial project. For now, amodem serves as a reference for high-performance audio modem implementations.

## Acknowledgments

- GGWave by Georgi Gerganov
- argparse by Pranav Srinivas Kumar
- doctest by Viktor Kirilov
- amodem by Roman Zeyde (reference implementation for OFDM-based audio modems)