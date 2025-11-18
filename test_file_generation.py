"""Test file generation endpoint directly."""

import requests
import base64
import json

def test_file_generation():
    """Test the /generate endpoint with file data."""
    
    # Create test file data
    test_data = b"This is test file data for audio generation"
    test_filename = "test.txt"
    
    # Encode as base64
    file_data_b64 = base64.b64encode(test_data).decode('utf-8')
    
    # Prepare request
    payload = {
        'file_data': file_data_b64,
        'filename': test_filename
    }
    
    print("Testing /generate endpoint with file data...")
    print(f"Payload size: {len(json.dumps(payload))} bytes")
    print(f"File size: {len(test_data)} bytes")
    
    try:
        response = requests.post(
            'http://localhost:5000/generate',
            json=payload,
            timeout=30
        )
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"\nResponse body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("\n[SUCCESS] File generation worked!")
        else:
            print(f"\n[FAIL] Server returned error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to server. Is it running?")
        print("Start the server with: python app.py")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_file_generation()

