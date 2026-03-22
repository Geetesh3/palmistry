import requests
import json

BASE_URL = "https://palmistry-fk4f.onrender.com"

def test_endpoint(name, method, path, data=None, files=None):
    print(f"Testing {name} ({path})...")
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{path}", timeout=30)
        else:
            if files:
                response = requests.post(f"{BASE_URL}{path}", files=files, timeout=30)
            else:
                response = requests.post(f"{BASE_URL}{path}", json=data, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ SUCCESS: {response.json()}")
        else:
            print(f"❌ FAILED ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"⚠️ ERROR: {str(e)}")
    print("-" * 50)

# 1. Test Root Status
test_endpoint("Root Status", "GET", "/")

# 2. Test Horoscope
test_endpoint("Daily Horoscope", "GET", "/get_horoscope?sign=Aries")

# 3. Test Kundli Generation
kundli_payload = {
    "name": "Test User",
    "dob": "1990-01-01",
    "time": "12:00",
    "lat": 28.61,
    "lon": 77.20
}
test_endpoint("Generate Kundli", "POST", "/generate_kundli", data=kundli_payload)

# 4. Test Palm Analysis (Requires a fake image)
print("Testing Palm Analysis (Simulated Image)...")
try:
    # Create a small dummy image
    import cv2
    import numpy as np
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("dummy.jpg", dummy_img)
    
    files = {'image': open('dummy.jpg', 'rb')}
    test_endpoint("Palm Analyze", "POST", "/analyze", files=files)
except Exception as e:
    print(f"Could not test image upload: {e}")
