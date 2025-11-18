"""Quick test to verify the system is working."""

import sys

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    try:
        import numpy as np
        print("[OK] numpy")
        
        from config import config
        print("[OK] config")
        
        from database import get_database
        print("[OK] database")
        
        from audio_engine import get_audio_engine
        print("[OK] audio_engine")
        
        from data_handler import process_text_data
        print("[OK] data_handler")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_database():
    """Test database operations."""
    print("\nTesting database...")
    try:
        from database import get_database
        import numpy as np
        
        db = get_database()
        print("[OK] Database initialized")
        
        # Test saving a transmission
        test_data = b"Hello, test!"
        test_waveform = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        test_metadata = {"test": True}
        
        transmission_id = db.save_transmission(
            input_type='text',
            input_data=test_data,
            waveform_data=test_waveform,
            metadata=test_metadata
        )
        print(f"[OK] Saved transmission with ID: {transmission_id}")
        
        # Test retrieving history
        history = db.get_history(limit=1)
        print(f"[OK] Retrieved history: {len(history)} entries")
        
        # Test getting by ID
        transmission = db.get_transmission_by_id(transmission_id)
        if transmission:
            print(f"[OK] Retrieved transmission by ID")
        else:
            print("[FAIL] Failed to retrieve transmission by ID")
            return False
        
        # Test delete (cleanup)
        cursor = db.conn.cursor()
        cursor.execute('DELETE FROM transmissions WHERE id = ?', (transmission_id,))
        db.conn.commit()
        print("[OK] Deleted test transmission")
        
        return True
    except Exception as e:
        print(f"[FAIL] Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_engine():
    """Test audio engine."""
    print("\nTesting audio engine...")
    try:
        from audio_engine import get_audio_engine
        
        engine = get_audio_engine()
        print(f"[OK] Audio engine initialized (backend: {engine.encoder_backend})")
        
        # Test encoding
        test_data = b"Test encoding"
        waveform = engine.encode_data(test_data)
        print(f"[OK] Encoded data to waveform (shape: {waveform.shape})")
        
        # Test status
        status = engine.get_status()
        print(f"[OK] Got engine status: {status}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Audio engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("GYGES System Test")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Database", test_database()))
    results.append(("Audio Engine", test_audio_engine()))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + ("=" * 50))
    if all_passed:
        print("ALL TESTS PASSED")
        print("System is ready to run!")
        print("\nTo start the application:")
        print("  python app.py")
    else:
        print("SOME TESTS FAILED")
        print("Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()

