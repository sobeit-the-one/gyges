"""Configuration management for the audio data transmitter."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Audio playback configuration."""
    sample_rate: int = 48000
    channels: int = 1
    chunk_size: int = 1024
    audio_device: Optional[int] = None  # None = default device


@dataclass
class GGWaveConfig:
    """GGWave encoding configuration."""
    protocol_id: int = 1  # Default protocol
    volume: int = 50  # Volume level (0-100)


@dataclass
class WebConfig:
    """Web server configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_folder: str = "uploads"
    database_path: str = "gyges.db"
    audio: AudioConfig = None
    ggwave: GGWaveConfig = None
    web: WebConfig = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.audio is None:
            self.audio = AudioConfig()
        if self.ggwave is None:
            self.ggwave = GGWaveConfig()
        if self.web is None:
            self.web = WebConfig()
        
        # Ensure upload folder exists (for temporary use if needed)
        os.makedirs(self.upload_folder, exist_ok=True)


# Global configuration instance
config = AppConfig()

