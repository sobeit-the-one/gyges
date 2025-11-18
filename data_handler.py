"""Data handling for text and binary file processing."""

import os
from typing import Union, Tuple
from config import config


def process_text_data(text: str) -> bytes:
    """
    Process text input and convert to bytes for encoding.
    
    Args:
        text: Text string to process
        
    Returns:
        Bytes representation of the text
    """
    if not text:
        raise ValueError("Text input cannot be empty")
    
    return text.encode('utf-8')


def process_binary_file(file_path: str) -> Tuple[bytes, str]:
    """
    Process binary file and prepare for encoding.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Tuple of (file_bytes, filename)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is too large
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    if file_size > config.max_file_size:
        raise ValueError(
            f"File size ({file_size} bytes) exceeds maximum "
            f"allowed size ({config.max_file_size} bytes)"
        )
    
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    filename = os.path.basename(file_path)
    return file_bytes, filename


def chunk_data(data: bytes, chunk_size: int = 1024) -> list:
    """
    Split data into chunks for transmission.
    
    Args:
        data: Data bytes to chunk
        chunk_size: Size of each chunk in bytes
        
    Returns:
        List of data chunks
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def detect_data_type(data: Union[str, bytes]) -> str:
    """
    Detect if data is text or binary.
    
    Args:
        data: Data to analyze (string or bytes)
        
    Returns:
        'text' or 'binary'
    """
    if isinstance(data, str):
        return 'text'
    
    # Try to decode as UTF-8 to check if it's text
    try:
        data.decode('utf-8')
        return 'text'
    except UnicodeDecodeError:
        return 'binary'

