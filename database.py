"""Database module for persistent transmission history."""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np

from config import config

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for storing transmission history."""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        self.db_path = db_path or config.database_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database and create table if it doesn't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transmissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    input_data BLOB NOT NULL,
                    waveform_data BLOB,
                    metadata TEXT
                )
            ''')
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_transmission(
        self,
        input_type: str,
        input_data: bytes,
        waveform_data: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a transmission to the database.
        
        Args:
            input_type: 'text' or 'file'
            input_data: Input data as bytes
            waveform_data: Encoded audio samples as numpy array
            metadata: Additional metadata (filename, size, etc.)
            
        Returns:
            ID of the saved transmission
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Convert waveform numpy array to bytes
            waveform_bytes = None
            if waveform_data is not None:
                waveform_bytes = waveform_data.tobytes()
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO transmissions 
                (timestamp, input_type, input_data, waveform_data, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, input_type, input_data, waveform_bytes, metadata_json))
            
            self.conn.commit()
            transmission_id = cursor.lastrowid
            logger.info(f"Saved transmission {transmission_id} to database")
            return transmission_id
        except Exception as e:
            logger.error(f"Failed to save transmission: {e}")
            raise
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve transmission history.
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of transmission dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, input_type, 
                       LENGTH(input_data) as input_size,
                       LENGTH(waveform_data) as waveform_size,
                       metadata
                FROM transmissions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            history = []
            
            for row in rows:
                # Parse metadata
                metadata = None
                if row['metadata']:
                    try:
                        metadata = json.loads(row['metadata'])
                    except json.JSONDecodeError:
                        metadata = {}
                
                # Get preview of input data
                input_preview = None
                if row['input_type'] == 'text':
                    cursor.execute('SELECT input_data FROM transmissions WHERE id = ?', (row['id'],))
                    input_bytes = cursor.fetchone()['input_data']
                    try:
                        text = input_bytes.decode('utf-8')
                        input_preview = text[:100] + ('...' if len(text) > 100 else '')
                    except UnicodeDecodeError:
                        input_preview = f"<binary data, {row['input_size']} bytes>"
                else:
                    if metadata and 'filename' in metadata:
                        input_preview = metadata['filename']
                    else:
                        input_preview = f"<file, {row['input_size']} bytes>"
                
                history.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'input_type': row['input_type'],
                    'input_preview': input_preview,
                    'input_size': row['input_size'],
                    'waveform_size': row['waveform_size'],
                    'metadata': metadata
                })
            
            return history
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            raise
    
    def get_transmission_by_id(self, transmission_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific transmission by ID.
        
        Args:
            transmission_id: ID of the transmission
            
        Returns:
            Transmission dictionary with full data, or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, input_type, input_data, 
                       waveform_data, metadata
                FROM transmissions
                WHERE id = ?
            ''', (transmission_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse metadata
            metadata = None
            if row['metadata']:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    metadata = {}
            
            # Convert waveform bytes back to numpy array if present
            waveform_data = None
            if row['waveform_data']:
                waveform_data = np.frombuffer(row['waveform_data'], dtype=np.float32)
            
            return {
                'id': row['id'],
                'timestamp': row['timestamp'],
                'input_type': row['input_type'],
                'input_data': row['input_data'],
                'waveform_data': waveform_data,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Failed to get transmission {transmission_id}: {e}")
            raise
    
    def clear_history(self) -> int:
        """
        Clear all transmission history.
        
        Returns:
            Number of rows deleted
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM transmissions')
            deleted_count = cursor.rowcount
            self.conn.commit()
            logger.info(f"Cleared {deleted_count} transmissions from database")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise
    
    def find_matching_transmission(self, data: bytes) -> Optional[int]:
        """
        Find transmission with matching input data.
        
        Args:
            data: Data to search for
            
        Returns:
            ID of matching transmission, or None
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id FROM transmissions
            WHERE input_data = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (data,))
        
        row = cursor.fetchone()
        return row[0] if row else None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance

