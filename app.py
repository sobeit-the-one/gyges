"""Main Flask application for the audio data transmitter."""

import os
import logging
import base64
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import threading
import numpy as np

from config import config
from audio_engine import get_audio_engine
from data_handler import process_text_data
from database import get_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Increase max upload size to 100MB for large WAV files
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = config.upload_folder

# Initialize audio engine and database
audio_engine = None
db = None
transmission_status = {"status": "idle", "message": ""}


def init_audio_engine():
    """Initialize the audio engine."""
    global audio_engine
    try:
        audio_engine = get_audio_engine()
        audio_engine.set_status_callback(update_transmission_status)
        logger.info("Audio engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audio engine: {e}")
        transmission_status["status"] = "error"
        transmission_status["message"] = f"Audio engine initialization failed: {str(e)}"


def init_database():
    """Initialize the database."""
    global db
    try:
        db = get_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def update_transmission_status(status: str):
    """Update transmission status."""
    transmission_status["status"] = status
    if status == "completed":
        transmission_status["message"] = "Transmission completed successfully"
    elif status.startswith("error"):
        transmission_status["message"] = status


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate audio waveform and save to database without playing."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get encoder selection and parameters from request
        selected_encoder = data.get('encoder', 'auto')
        fsk_params = data.get('fsk_params', {})
        ggwave_params = data.get('ggwave_params', {})
        
        input_type = None
        input_bytes = None
        metadata = {}
        
        # Check if text or file data
        if 'text' in data:
            # Process text data
            text = data['text']
            if not text:
                return jsonify({"error": "Text cannot be empty"}), 400
            
            try:
                input_bytes = process_text_data(text)
                input_type = 'text'
                # Use timestamp for unique text filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                metadata = {
                    'text_length': len(text),
                    'filename': f'gyges_text_{timestamp}.txt'
                }
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        
        elif 'file_data' in data:
            # File data is already in bytes (from upload)
            try:
                file_data_b64 = data['file_data']
                input_bytes = base64.b64decode(file_data_b64)
                input_type = 'file'
                metadata = {
                    'filename': data.get('filename', 'unknown.bin'),
                    'file_size': len(input_bytes)
                }
            except Exception as e:
                return jsonify({"error": f"Invalid file data: {str(e)}"}), 400
        else:
            return jsonify({"error": "Must provide either 'text' or 'file_data'"}), 400
        
        # Generate audio waveform
        if not audio_engine:
            return jsonify({"error": "Audio engine not initialized"}), 500
        
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
        
        # Encode data to audio (without playing), forcing encoder if not 'auto'
        force_encoder = None if selected_encoder == 'auto' else selected_encoder
        waveform_data, actual_encoder = audio_engine.encode_data(
            input_bytes, 
            force_encoder=force_encoder,
            fsk_params=fsk_params,
            ggwave_params=ggwave_params
        )
        
        # Save to database with actual encoder used
        metadata['encoder_backend'] = actual_encoder
        metadata['encoder_selected'] = selected_encoder
        metadata['data_size'] = len(input_bytes)
        
        transmission_id = db.save_transmission(
            input_type=input_type,
            input_data=input_bytes,
            waveform_data=waveform_data,
            metadata=metadata
        )
        
        return jsonify({
            "success": True,
            "message": "Audio waveform generated and saved",
            "transmission_id": transmission_id,
            "data_size": len(input_bytes)
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in generate endpoint: {e}\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500


@app.route('/play/<int:transmission_id>', methods=['POST'])
def play_transmission(transmission_id):
    """Play a stored transmission from the database."""
    global transmission_status
    
    try:
        if not audio_engine:
            return jsonify({"error": "Audio engine not initialized"}), 500
        
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
        
        # Get transmission from database
        transmission = db.get_transmission_by_id(transmission_id)
        if not transmission:
            return jsonify({"error": "Transmission not found"}), 404
        
        waveform_data = transmission['waveform_data']
        if waveform_data is None:
            return jsonify({"error": "No waveform data available for this transmission"}), 400
        
        # Ensure it's a numpy array
        if not isinstance(waveform_data, np.ndarray):
            waveform_data = np.array(waveform_data, dtype=np.float32)
        
        # Update status
        transmission_status["status"] = "transmitting"
        transmission_status["message"] = "Playing transmission..."
        
        # Play audio in background thread
        def play_thread():
            try:
                audio_engine.transmit_from_waveform(waveform_data, blocking=True)
            except Exception as e:
                logger.error(f"Playback error: {e}")
                update_transmission_status(f"error: {str(e)}")
        
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Playback started",
            "transmission_id": transmission_id
        })
        
    except Exception as e:
        logger.error(f"Error in play endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload - stores file data in memory for transmission."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read file data into memory
        file_data = file.read()
        file_size = len(file_data)
        
        # Check file size
        if file_size > config.max_file_size:
            return jsonify({"error": f"File size ({file_size} bytes) exceeds maximum allowed size ({config.max_file_size} bytes)"}), 400
        
        # Return file data as base64 for frontend to send in generate request
        file_data_b64 = base64.b64encode(file_data).decode('utf-8')
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "file_data": file_data_b64,
            "file_size": file_size
        })
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Get transmission status."""
    status_info = transmission_status.copy()
    
    if audio_engine:
        engine_status = audio_engine.get_status()
        status_info.update(engine_status)
    
    return jsonify(status_info)


@app.route('/encoders', methods=['GET'])
def get_encoders():
    """Get list of available encoders."""
    try:
        encoders = ['auto', 'simple_fsk']
        current_backend = 'unknown'
        
        if audio_engine:
            current_backend = audio_engine.encoder_backend
            logger.info(f"Current encoder backend: {current_backend}")
            
            # Add available backends as options
            if audio_engine.encoder_backend == 'ggwave_cpp':
                encoders.append('ggwave')
            
        else:
            logger.warning("audio_engine is None in /encoders endpoint")
        
        return jsonify({
            'success': True,
            'encoders': encoders,
            'current_backend': current_backend,
            'default': audio_engine.encoder_backend if audio_engine else 'simple_fsk'
        })
    except Exception as e:
        logger.error(f"Error getting encoders: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_transmission():
    """Stop current transmission."""
    try:
        if audio_engine:
            audio_engine.stop()
            update_transmission_status("stopped")
            return jsonify({"success": True, "message": "Transmission stopped"})
        else:
            return jsonify({"error": "Audio engine not initialized"}), 500
    except Exception as e:
        logger.error(f"Error stopping transmission: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    """Get transmission history from database."""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
        
        limit = request.args.get('limit', 50, type=int)
        history_data = db.get_history(limit=limit)
        
        return jsonify({
            "success": True,
            "history": history_data
        })
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/delete/<int:transmission_id>', methods=['DELETE'])
def delete_transmission(transmission_id):
    """Delete a transmission from the database."""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
        
        # Check if transmission exists
        transmission = db.get_transmission_by_id(transmission_id)
        if not transmission:
            return jsonify({"error": "Transmission not found"}), 404
        
        # Delete from database
        cursor = db.conn.cursor()
        cursor.execute('DELETE FROM transmissions WHERE id = ?', (transmission_id,))
        db.conn.commit()
        
        deleted_count = cursor.rowcount
        
        return jsonify({
            "success": True,
            "message": f"Deleted transmission {transmission_id}",
            "deleted_count": deleted_count
        })
    except Exception as e:
        logger.error(f"Error deleting transmission: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear all transmission history."""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
        
        deleted_count = db.clear_history()
        
        return jsonify({
            "success": True,
            "message": f"Cleared {deleted_count} transmission(s)",
            "deleted_count": deleted_count
        })
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/download-wav/<int:transmission_id>', methods=['GET'])
def download_wav(transmission_id):
    """Download waveform as WAV file."""
    try:
        transmission = db.get_transmission_by_id(transmission_id)
        
        if not transmission:
            return jsonify({'success': False, 'error': 'Transmission not found'}), 404
        
        waveform_data = transmission['waveform_data']
        
        if waveform_data is None:
            return jsonify({'success': False, 'error': 'No waveform data'}), 404
        
        # Export to WAV
        wav_bytes = audio_engine.export_to_wav(waveform_data)
        
        # Create filename
        timestamp = transmission['timestamp'].replace(':', '-').replace(' ', '_')
        filename = f"gyges_{transmission_id}_{timestamp}.wav"
        
        return send_file(
            io.BytesIO(wav_bytes),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading WAV: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download-data/<int:transmission_id>', methods=['GET'])
def download_data(transmission_id):
    """Download original input data."""
    try:
        transmission = db.get_transmission_by_id(transmission_id)
        
        if not transmission:
            return jsonify({'success': False, 'error': 'Transmission not found'}), 404
        
        input_data = transmission['input_data']
        input_type = transmission['input_type']
        
        metadata = transmission.get('metadata', {})
        
        # Handle decoded entries
        if input_type in ['decoded_text', 'text']:
            # Get original filename or generate one
            filename = metadata.get('filename', f'gyges_{transmission_id}.txt')
            if 'decoded' in input_type and not filename.endswith('_decoded.txt'):
                # Add _decoded before extension
                base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'txt')
                filename = f'{base}_decoded.{ext}'
            
            return send_file(
                io.BytesIO(input_data),
                mimetype='text/plain',
                as_attachment=True,
                download_name=filename
            )
        elif input_type == 'decoded_binary':
            # For decoded binary, preserve original filename extension
            original_filename = metadata.get('filename', f'gyges_{transmission_id}_decoded.bin')
            # Strip .wav extension if present
            if original_filename.endswith('.wav'):
                original_filename = original_filename[:-4]
            # Add _decoded before extension
            if '.' in original_filename:
                base, ext = original_filename.rsplit('.', 1)
                filename = f'{base}_decoded.{ext}'
            else:
                filename = f'{original_filename}_decoded.bin'
            
            return send_file(
                io.BytesIO(input_data),
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=filename
            )
        else:
            # Regular file transmission - use stored filename
            filename = metadata.get('filename', f'gyges_{transmission_id}.bin')
            
            return send_file(
                io.BytesIO(input_data),
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=filename
            )
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/decode', methods=['POST'])
def decode():
    """Decode audio file back to data."""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio_file']
        
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Get decoder selection and FSK parameters
        decoder_method = request.form.get('decoder', 'auto')
        custom_filename = request.form.get('output_filename', '').strip()
        
        # Get FSK decode parameters from form
        fsk_params = {}
        if request.form.get('fsk_bit_duration'):
            fsk_params['bit_duration'] = float(request.form.get('fsk_bit_duration')) / 1000.0  # ms to seconds
        if request.form.get('fsk_freq_0'):
            fsk_params['freq_0'] = int(request.form.get('fsk_freq_0'))
        if request.form.get('fsk_freq_1'):
            fsk_params['freq_1'] = int(request.form.get('fsk_freq_1'))
        
        # Get GGWave decode parameters from form
        ggwave_params = {}
        if request.form.get('ggwave_sample_rate'):
            ggwave_params['ggwave_sample_rate'] = int(request.form.get('ggwave_sample_rate'))
        
        # Read audio file
        audio_bytes = audio_file.read()
        
        # Import from WAV
        audio_samples, sample_rate = audio_engine.import_from_wav(audio_bytes)
        
        # Decode based on selected method
        if decoder_method == 'ggwave':
            decoded_data, decode_info = audio_engine.decode_ggwave(audio_samples, ggwave_params=ggwave_params)
        elif decoder_method == 'simple_fsk':
            decoded_data, decode_info = audio_engine.decode_simple_fsk(
                audio_samples,
                bit_duration=fsk_params.get('bit_duration', 0.01),
                freq_0=fsk_params.get('freq_0', 1200),
                freq_1=fsk_params.get('freq_1', 1800)
            )
        else:
            # Auto-detect: try all decoders
            decoded_data, decode_info = audio_engine.decode_with_auto_detect(
                audio_samples, 
                fsk_params=fsk_params,
                ggwave_params=ggwave_params
            )
        
        if decoded_data is None:
            return jsonify({
                'success': False,
                'error': decode_info.get('error', 'Decoding failed'),
                'details': decode_info
            }), 400
        
        # Determine if text or binary
        try:
            decoded_text = decoded_data.decode('utf-8')
            is_text = True
            preview = decoded_text[:100]
        except UnicodeDecodeError:
            is_text = False
            preview = f"Binary data ({len(decoded_data)} bytes)"
        
        # Determine output filename
        if custom_filename:
            # User provided custom filename
            output_filename = custom_filename
        elif is_text:
            # Auto-generate .txt filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'gyges_decoded_{timestamp}.txt'
        else:
            # Auto-generate .bin filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'gyges_decoded_{timestamp}.bin'
        
        # Save to database with decoder method used
        metadata = {
            'filename': output_filename,  # Use determined filename
            'original_wav': audio_file.filename,  # Save original WAV name for reference
            'decoded': True,
            'decode_info': decode_info,
            'encoder_backend': decode_info.get('encoder', decoder_method),  # Save which decoder was used
            'decoder_method': decoder_method,  # Save what user selected
            'sample_rate': sample_rate
        }
        
        transmission_id = db.save_transmission(
            input_type='decoded_text' if is_text else 'decoded_binary',
            input_data=decoded_data,
            waveform_data=audio_samples,
            metadata=metadata
        )
        
        # Check for matches
        match_id = db.find_matching_transmission(decoded_data)
        if match_id and match_id != transmission_id:
            decode_info['matched_transmission_id'] = match_id
        
        return jsonify({
            'success': True,
            'transmission_id': transmission_id,
            'is_text': is_text,
            'preview': preview,
            'decode_info': decode_info,
            'data_size': len(decoded_data)
        })
        
    except Exception as e:
        logger.error(f"Error in decode endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'details': traceback.format_exc()
        }), 500


def init_database():
    """Initialize the database."""
    global db
    try:
        db = get_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


if __name__ == '__main__':
    # Initialize database on startup
    init_database()
    
    # Initialize audio engine on startup
    init_audio_engine()
    
    # Run Flask app
    app.run(
        host=config.web.host,
        port=config.web.port,
        debug=config.web.debug
    )

