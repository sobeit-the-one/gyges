"""Audio engine for data-over-sound encoding and playback with multiple backend support."""

import base64
import threading
import queue
import logging
import io
import wave
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np
from scipy import signal as scipy_signal

# Try to import audio library
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    try:
        import sounddevice as sd
        PYAUDIO_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("PyAudio not available, using sounddevice as fallback")
    except ImportError:
        raise ImportError("Neither pyaudio nor sounddevice is available. Please install one: pip install pyaudio or pip install sounddevice")

# Encoder backend detection
ENCODER_BACKEND = None

# Check for gyges_cpp.exe binary (for ggwave encoding)
GYGES_CPP_PATH = None
possible_paths = [
    Path(__file__).parent / "bin" / "Release" / "gyges_cpp.exe",
    Path(__file__).parent / "bin" / "Debug" / "gyges_cpp.exe",
    Path(__file__).parent / "bin" / "MinSizeRel" / "gyges_cpp.exe",
]

for path in possible_paths:
    if path.exists():
        GYGES_CPP_PATH = path
        ENCODER_BACKEND = "ggwave_cpp"
        logger = logging.getLogger(__name__)
        logger.info(f"Found gyges_cpp binary at {path}")
        break

if ENCODER_BACKEND is None:
    # Fallback to simple FSK encoding
    ENCODER_BACKEND = "simple_fsk"
    logger = logging.getLogger(__name__)
    logger.warning("No gyges_cpp binary found. Using simple FSK encoding only.")

from config import config

logger = logging.getLogger(__name__)

# Amodem removed due to unreliable API
AMODEM_AVAILABLE = False

# Audio conversion constants
INT16_MAX = 32767.0  # Maximum positive value for signed 16-bit integer
INT16_SCALE = 32768.0  # Scale factor for int16 ↔ float32 conversion (2^15)


class AudioEngine:
    """Handles data-over-sound encoding and audio playback with multiple backends."""
    
    def __init__(self):
        """Initialize the audio engine."""
        self.encoder_instance = None
        self.encoder_backend = ENCODER_BACKEND
        self.audio_stream = None
        self.pyaudio_instance = None
        self.transmission_queue = queue.Queue()
        self.is_playing = False
        self.current_transmission = None
        self.playback_thread = None
        self.status_callback: Optional[Callable] = None
        
        self._initialize_encoder()
        self._initialize_audio()
    
    def _initialize_encoder(self):
        """Initialize the encoding backend."""
        try:
            if self.encoder_backend == "ggwave_cpp":
                # gyges_cpp binary doesn't need initialization, just verify it exists
                if not GYGES_CPP_PATH or not GYGES_CPP_PATH.exists():
                    raise RuntimeError("gyges_cpp binary not found")
                self.encoder_instance = True
                logger.info(f"Using gyges_cpp binary at {GYGES_CPP_PATH}")
            elif self.encoder_backend == "simple_fsk":
                # Simple FSK doesn't need instance
                self.encoder_instance = True
                logger.info("Simple FSK encoder ready (basic fallback)")
            else:
                raise RuntimeError(f"Unknown encoder backend: {self.encoder_backend}")
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            raise
    
    def _initialize_audio(self):
        """Initialize audio playback."""
        try:
            if PYAUDIO_AVAILABLE:
                self.pyaudio_instance = pyaudio.PyAudio()
                logger.info("PyAudio initialized successfully")
            else:
                # sounddevice doesn't need instance
                logger.info("sounddevice ready for playback")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def encode_data(self, data: bytes, force_encoder: str = None, fsk_params: dict = None, ggwave_params: dict = None) -> Tuple[np.ndarray, str]:
        """
        Encode data to audio samples using the available backend.
        
        Args:
            data: Data bytes to encode
            force_encoder: Force specific encoder ('simple_fsk', 'ggwave')
            fsk_params: FSK parameters (bit_duration, freq_0, freq_1)
            ggwave_params: GGWave parameters (protocol, volume)
            
        Returns:
            Tuple of (audio_samples, actual_encoder_used)
        """
        if self.encoder_instance is None:
            raise RuntimeError("Encoder not initialized")
        
        # Allow forcing a specific encoder
        encoder_to_use = force_encoder if force_encoder else self.encoder_backend
        
        try:
            if encoder_to_use == "simple_fsk":
                return self._encode_simple_fsk(data, fsk_params), "simple_fsk"
            elif encoder_to_use == "ggwave_cpp" or encoder_to_use == "ggwave":
                return self._encode_ggwave_cpp(data, ggwave_params)
            elif self.encoder_backend == "ggwave_cpp":
                return self._encode_ggwave_cpp(data, ggwave_params)
            elif self.encoder_backend == "simple_fsk":
                return self._encode_simple_fsk(data, fsk_params), "simple_fsk"
            else:
                raise RuntimeError(f"Unknown encoder backend: {encoder_to_use}")
        except Exception as e:
            logger.error(f"Failed to encode data: {e}")
            raise
    
    def _encode_ggwave_cpp(self, data: bytes, ggwave_params: dict = None) -> Tuple[np.ndarray, str]:
        """Encode using gyges_cpp binary.
        
        Returns:
            Tuple of (audio_samples, encoder_used)
        """
        if ggwave_params is None:
            ggwave_params = {}
        
        # ggwave has a hardcoded 140-byte limit
        GGWAVE_MAX_BYTES = 140
        
        # Check size before encoding
        if len(data) > GGWAVE_MAX_BYTES:
            raise ValueError(f"Data too large for GGWave: {len(data)} bytes (max {GGWAVE_MAX_BYTES} bytes). Please use FSK encoder instead.")
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.dat') as input_file:
            input_file.write(data)
            input_path = input_file.name
        
        output_path = input_path.replace('.dat', '.wav')
        
        try:
            # Build command
            protocol = ggwave_params.get('protocol', config.ggwave.protocol_id)
            volume = ggwave_params.get('volume', config.ggwave.volume)
            sample_rate = config.audio.sample_rate
            
            cmd = [
                str(GYGES_CPP_PATH),
                input_path,
                '--method', 'ggwave',
                '--output', output_path,
                '--protocol', str(protocol),
                '--volume', str(volume),
                '--sample-rate', str(sample_rate)
            ]
            
            logger.info(f"Running gyges_cpp: {' '.join(cmd)}")
            
            # Run the binary
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"gyges_cpp output: {result.stdout}")
            
            # Read the generated WAV file as bytes
            with open(output_path, 'rb') as f:
                wav_bytes = f.read()
            
            audio_samples, _ = self.import_from_wav(wav_bytes)
            
            return audio_samples, "ggwave_cpp"
            
        except subprocess.CalledProcessError as e:
            logger.error(f"gyges_cpp failed: {e.stderr}")
            raise RuntimeError(f"GGWave encoding failed: {e.stderr}")
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {e}")
    
    
    def _encode_simple_fsk(self, data: bytes, fsk_params: dict = None) -> np.ndarray:
        """Simple FSK encoding fallback (basic but always works)."""
        if fsk_params is None:
            fsk_params = {}
        
        # Calculate max data size based on bit duration to keep WAV < 50MB
        # Audio size = data_bytes * 8 bits * bit_duration * sample_rate * 2 bytes/sample
        sample_rate = config.audio.sample_rate
        bit_duration = fsk_params.get('bit_duration', 0.01)
        max_wav_size = 50 * 1024 * 1024  # 50MB limit for WAV
        max_bytes = int(max_wav_size / (8 * bit_duration * sample_rate * 2))
        
        if len(data) > max_bytes:
            estimated_wav_mb = (len(data) * 8 * bit_duration * sample_rate * 2) / (1024 * 1024)
            raise ValueError(
                f"Data too large for FSK: {len(data)} bytes would create ~{estimated_wav_mb:.1f}MB WAV (max 50MB). "
                f"Max data size at {bit_duration*1000:.0f}ms bit duration: {max_bytes} bytes. "
                f"Reduce bit duration to {(bit_duration * max_bytes / len(data) * 1000):.1f}ms or reduce file size."
            )
        
        # Convert data to binary string
        binary_str = ''.join(format(byte, '08b') for byte in data)
        
        # FSK parameters - use custom or defaults
        sample_rate = config.audio.sample_rate
        bit_duration = fsk_params.get('bit_duration', 0.01)  # seconds
        samples_per_bit = int(sample_rate * bit_duration)
        
        # Frequencies for 0 and 1
        freq_0 = fsk_params.get('freq_0', 1200)  # Hz
        freq_1 = fsk_params.get('freq_1', 1800)  # Hz
        
        # Generate audio samples
        total_samples = len(binary_str) * samples_per_bit
        audio_samples = np.zeros(total_samples, dtype=np.float32)
        
        # Generate time vector for one bit
        t_bit = np.linspace(0, bit_duration, samples_per_bit, False)
        
        for i, bit in enumerate(binary_str):
            start_idx = i * samples_per_bit
            end_idx = start_idx + samples_per_bit
            freq = freq_1 if bit == '1' else freq_0
            audio_samples[start_idx:end_idx] = np.sin(2 * np.pi * freq * t_bit)
        
        # Normalize and apply volume
        if audio_samples.max() > 0:
            audio_samples = audio_samples / audio_samples.max()
        audio_samples = audio_samples * (config.ggwave.volume / 100.0)
        
        logger.info(f"FSK encoded {len(data)} bytes to {len(audio_samples)} samples ({len(audio_samples) * 4 / 1024 / 1024:.2f} MB)")
        
        return audio_samples
    
    def encode_with_header_and_crc(self, data: bytes, encoder_type: str = 'simple_fsk') -> np.ndarray:
        """
        Encode data with header and CRC for reliable transmission.
        
        Args:
            data: Data bytes to encode
            encoder_type: Type of encoder being used
            
        Returns:
            Audio samples with header prepended and CRC appended
        """
        # Add CRC32 checksum to data
        crc = zlib.crc32(data)
        data_with_crc = data + struct.pack('<I', crc)
        
        # Create header
        header = self._create_transmission_header(len(data), encoder_type)
        
        # Encode header and data separately
        header_audio = self.encode_data(header)
        data_audio = self.encode_data(data_with_crc)
        
        # Concatenate with small gap
        gap_samples = int(0.1 * config.audio.sample_rate)  # 100ms gap
        gap = np.zeros(gap_samples, dtype=np.float32)
        
        full_audio = np.concatenate([header_audio, gap, data_audio])
        
        return full_audio
    
    def _create_transmission_header(self, data_length: int, encoder_type: str) -> bytes:
        """Create transmission header with metadata."""
        header = bytearray()
        
        # Magic bytes "GYGES"
        header.extend(b'GYGES')
        
        # Encoder type (1 byte)
        encoder_map = {'simple_fsk': 0, 'ggwave_cpp': 1}
        header.append(encoder_map.get(encoder_type, 0))
        
        # FSK parameters (if FSK)
        if encoder_type == 'simple_fsk':
            # Bit duration in microseconds (2 bytes)
            bit_duration_us = int(0.01 * 1_000_000)  # 10ms in microseconds
            header.extend(struct.pack('<H', bit_duration_us))
            
            # Frequencies (2 bytes each)
            header.extend(struct.pack('<H', 1200))  # freq_0
            header.extend(struct.pack('<H', 1800))  # freq_1
        
        # Data length (4 bytes)
        header.extend(struct.pack('<I', data_length))
        
        # Header CRC (2 bytes)
        header_crc = zlib.crc32(bytes(header)) & 0xFFFF
        header.extend(struct.pack('<H', header_crc))
        
        return bytes(header)
    
    def decode_ggwave(self, audio_samples: np.ndarray) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Decode using ggwave - NOT AVAILABLE (gyges_cpp binary only supports encoding)."""
        logger.warning("GGWave decoding not available with gyges_cpp binary")
        return None, {'error': 'GGWave decoding not available (gyges_cpp binary only supports encoding)'}
    
    
    def decode_simple_fsk(self, audio_samples: np.ndarray, 
                          bit_duration: float = 0.01,
                          freq_0: int = 1200, 
                          freq_1: int = 1800,
                          sample_rate: int = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Decode FSK audio back to data bytes.
        
        Args:
            audio_samples: Audio waveform to decode
            bit_duration: Duration of each bit in seconds
            freq_0: Frequency for bit '0' in Hz
            freq_1: Frequency for bit '1' in Hz
            sample_rate: Sample rate (default from config)
            
        Returns:
            Tuple of (decoded_bytes, decode_info)
            decode_info contains: confidence, snr, bit_errors, etc.
        """
        if sample_rate is None:
            sample_rate = config.audio.sample_rate
        
        logger.info(f"Starting FSK decode: {len(audio_samples)} samples, bit_duration={bit_duration}s, freq_0={freq_0}Hz, freq_1={freq_1}Hz")
        
        # Apply bandpass filter to reduce noise
        filtered_audio = self._apply_bandpass_filter(audio_samples, freq_0, freq_1, sample_rate)
        
        # Calculate samples per bit
        samples_per_bit = int(sample_rate * bit_duration)
        
        # Decode bits
        bits = []
        confidences = []
        
        num_bits = len(filtered_audio) // samples_per_bit
        
        for i in range(num_bits):
            start_idx = i * samples_per_bit
            end_idx = start_idx + samples_per_bit
            chunk = filtered_audio[start_idx:end_idx]
            
            # Use FFT to detect frequencies
            bit_value, confidence = self._detect_fsk_bit(chunk, freq_0, freq_1, sample_rate)
            bits.append(bit_value)
            confidences.append(confidence)
        
        # Convert bits to bytes (MSB first to match encoder)
        decoded_bytes = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte_bits = bits[i:i+8]
            byte_value = 0
            for j, bit in enumerate(byte_bits):
                byte_value |= (bit << (7 - j))  # MSB first
            decoded_bytes.append(byte_value)
        
        # Calculate statistics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        decode_info = {
            'confidence': float(avg_confidence),
            'bits_decoded': len(bits),
            'bytes_decoded': len(decoded_bytes),
            'sample_rate': sample_rate,
            'bit_duration': bit_duration,
            'freq_0': freq_0,
            'freq_1': freq_1,
            'encoder': 'simple_fsk'
        }
        
        logger.info(f"FSK decode complete: {len(decoded_bytes)} bytes, confidence={avg_confidence:.2%}")
        
        return bytes(decoded_bytes), decode_info
    
    def _apply_bandpass_filter(self, audio: np.ndarray, freq_0: int, freq_1: int, sample_rate: int) -> np.ndarray:
        """Apply bandpass filter centered on transmission frequencies."""
        # Design bandpass filter
        low_freq = freq_0 - 300
        high_freq = freq_1 + 300
        
        # Normalize frequencies
        nyquist = sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Design butterworth filter
        b, a = scipy_signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter
        filtered = scipy_signal.filtfilt(b, a, audio)
        
        return filtered
    
    def _detect_fsk_bit(self, chunk: np.ndarray, freq_0: int, freq_1: int, sample_rate: int) -> Tuple[int, float]:
        """
        Detect which frequency is present in audio chunk using FFT.
        
        Returns:
            Tuple of (bit_value, confidence)
        """
        # Apply window to reduce spectral leakage
        window = np.hanning(len(chunk))
        windowed = chunk * window
        
        # Compute FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1/sample_rate)
        power = np.abs(fft) ** 2
        
        # Find power at target frequencies
        freq_0_idx = np.argmin(np.abs(freqs - freq_0))
        freq_1_idx = np.argmin(np.abs(freqs - freq_1))
        
        # Get power in narrow bands around each frequency
        band_width = 3  # Check ±3 bins
        power_0 = np.sum(power[max(0, freq_0_idx-band_width):freq_0_idx+band_width+1])
        power_1 = np.sum(power[max(0, freq_1_idx-band_width):freq_1_idx+band_width+1])
        
        # Determine bit value
        bit_value = 1 if power_1 > power_0 else 0
        
        # Calculate confidence (ratio of stronger to weaker)
        total_power = power_0 + power_1
        if total_power > 0:
            confidence = max(power_0, power_1) / total_power
        else:
            confidence = 0.0
        
        return bit_value, confidence
    
    def decode_with_auto_detect(self, audio_samples: np.ndarray, 
                                  fsk_params: dict = None) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Decode audio with automatic decoder detection.
        Currently only supports FSK (ggwave decoding not available with gyges_cpp binary).
        
        Args:
            audio_samples: Audio samples to decode
            fsk_params: FSK parameters to use if FSK decoder is tried
        
        Returns:
            Tuple of (decoded_data, decode_info)
        """
        # Note: GGWave decoding not available with gyges_cpp binary
        # Only FSK decoding is supported
        
        # Try FSK with provided or default parameters
        try:
            if fsk_params is None:
                fsk_params = {}
            decoded, info = self.decode_simple_fsk(
                audio_samples,
                bit_duration=fsk_params.get('bit_duration', 0.01),
                freq_0=fsk_params.get('freq_0', 1200),
                freq_1=fsk_params.get('freq_1', 1800)
            )
            info['encoder'] = 'simple_fsk'
            return decoded, info
        except Exception as e:
            logger.error(f"FSK decode failed: {e}")
            return None, {'error': 'FSK decode failed'}
    
    
    def export_to_wav(self, audio_samples: np.ndarray, sample_rate: int = None) -> bytes:
        """
        Export numpy audio samples to WAV format bytes.
        
        Args:
            audio_samples: Audio data as numpy array
            sample_rate: Sample rate (default from config)
            
        Returns:
            WAV file as bytes
        """
        if sample_rate is None:
            sample_rate = config.audio.sample_rate
        
        # Convert float32 to int16
        if audio_samples.dtype == np.float32 or audio_samples.dtype == np.float64:
            # Ensure samples are in range [-1, 1]
            audio_samples = np.clip(audio_samples, -1.0, 1.0)
            # Convert to int16 (use INT16_SCALE for full dynamic range)
            audio_int16 = (audio_samples * INT16_SCALE).astype(np.int16)
        else:
            audio_int16 = audio_samples.astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_bytes = wav_buffer.getvalue()
        logger.info(f"Exported {len(audio_samples)} samples to WAV ({len(wav_bytes)} bytes)")
        
        return wav_bytes
    
    def import_from_wav(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Import audio samples from WAV format bytes.
        
        Args:
            wav_bytes: WAV file as bytes
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        wav_buffer = io.BytesIO(wav_bytes)
        
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            audio_bytes = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            elif sample_width == 1:  # 8-bit
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16) - 128
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to float32 [-1, 1]
            audio_float = audio_int16.astype(np.float32) / INT16_SCALE
            
            # Handle multi-channel audio (convert to mono)
            if n_channels > 1:
                audio_float = audio_float.reshape(-1, n_channels).mean(axis=1)
            
            logger.info(f"Imported WAV: {len(audio_float)} samples at {sample_rate}Hz")
            
            return audio_float, sample_rate
    
    def _playback_worker(self):
        """Worker thread for audio playback."""
        while self.is_playing or not self.transmission_queue.empty():
            try:
                # Get next audio data from queue
                audio_samples = self.transmission_queue.get(timeout=1.0)
                
                if audio_samples is None:  # Shutdown signal
                    break
                
                # Convert float32 to int16 for playback (with clipping to prevent overflow)
                audio_int16 = (np.clip(audio_samples, -1.0, 1.0) * INT16_SCALE).astype(np.int16)
                
                # Open audio stream if not already open
                if PYAUDIO_AVAILABLE:
                    if not self.audio_stream or not self.audio_stream.is_active():
                        self.audio_stream = self.pyaudio_instance.open(
                            format=pyaudio.paInt16,
                            channels=config.audio.channels,
                            rate=config.audio.sample_rate,
                            output=True,
                            output_device_index=config.audio.audio_device,
                            frames_per_buffer=config.audio.chunk_size
                        )
                
                # Play audio in chunks
                chunk_size = config.audio.chunk_size
                if PYAUDIO_AVAILABLE:
                    for i in range(0, len(audio_int16), chunk_size):
                        chunk = audio_int16[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            # Pad last chunk with zeros
                            padded = np.zeros(chunk_size, dtype=np.int16)
                            padded[:len(chunk)] = chunk
                            chunk = padded
                        
                        self.audio_stream.write(chunk.tobytes())
                else:
                    # Use sounddevice
                    import sounddevice as sd
                    sd.play(audio_samples, samplerate=config.audio.sample_rate)
                    sd.wait()
                
                self.transmission_queue.task_done()
                
                if self.status_callback:
                    self.status_callback("completed")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error during playback: {e}")
                if self.status_callback:
                    self.status_callback(f"error: {str(e)}")
        
        # Close audio stream
        if PYAUDIO_AVAILABLE and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
    
    def transmit_from_waveform(self, waveform_data: np.ndarray, blocking: bool = False):
        """
        Play audio from pre-encoded waveform data.
        
        Args:
            waveform_data: Pre-encoded audio samples as numpy array
            blocking: If True, wait for playback to complete
        """
        if waveform_data is None or len(waveform_data) == 0:
            raise ValueError("Waveform data cannot be empty")
        
        try:
            # Add to queue
            self.transmission_queue.put(waveform_data)
            
            # Start playback thread if not running
            if not self.is_playing:
                self.is_playing = True
                self.playback_thread = threading.Thread(
                    target=self._playback_worker,
                    daemon=True
                )
                self.playback_thread.start()
            
            if blocking:
                self.transmission_queue.join()
                
        except Exception as e:
            logger.error(f"Failed to play waveform: {e}")
            raise
    
    def transmit(self, data: bytes, blocking: bool = False) -> np.ndarray:
        """
        Transmit data as audio (legacy method - encodes and plays).
        
        Args:
            data: Data bytes to transmit
            blocking: If True, wait for transmission to complete
            
        Returns:
            Encoded audio samples as numpy array
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        try:
            # Encode data to audio
            audio_samples = self.encode_data(data)
            
            # Play it
            self.transmit_from_waveform(audio_samples, blocking=blocking)
            
            # Return waveform data for storage
            return audio_samples
                
        except Exception as e:
            logger.error(f"Failed to transmit data: {e}")
            raise
    
    def stop(self):
        """Stop current transmission."""
        self.is_playing = False
        # Clear queue
        while not self.transmission_queue.empty():
            try:
                self.transmission_queue.get_nowait()
            except queue.Empty:
                break
        
        if PYAUDIO_AVAILABLE and self.audio_stream:
            self.audio_stream.stop_stream()
    
    def get_status(self) -> dict:
        """
        Get current transmission status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_playing": self.is_playing,
            "queue_size": self.transmission_queue.qsize(),
            "has_active_stream": (PYAUDIO_AVAILABLE and self.audio_stream is not None and self.audio_stream.is_active()) if (PYAUDIO_AVAILABLE and self.audio_stream) else False,
            "encoder_backend": self.encoder_backend
        }
    
    def set_status_callback(self, callback: Callable):
        """Set callback function for status updates."""
        self.status_callback = callback
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if PYAUDIO_AVAILABLE:
            if self.audio_stream:
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
        # No cleanup needed for ggwave_cpp (binary-based)


# Global audio engine instance
_audio_engine: Optional[AudioEngine] = None


def get_audio_engine() -> AudioEngine:
    """Get or create the global audio engine instance."""
    global _audio_engine
    if _audio_engine is None:
        _audio_engine = AudioEngine()
    return _audio_engine

