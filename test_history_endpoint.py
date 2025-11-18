"""Test history endpoint."""

import requests
import json

def test_history():
    """Test the /history endpoint."""
    
    print("Testing /history endpoint...")
    
    try:
        response = requests.get(
            'http://localhost:5000/history?limit=50',
            timeout=10
        )
        
        print(f"\nResponse status: {response.status_code}")
        print(f"\nResponse body:")
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if data.get('success'):
            history = data.get('history', [])
            print(f"\n[SUCCESS] Found {len(history)} history entries")
            for entry in history:
                print(f"  - ID {entry['id']}: {entry['input_type']} - {entry['input_preview'][:50]}")
        else:
            print(f"\n[FAIL] History endpoint failed")
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to server. Is it running?")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_history()

