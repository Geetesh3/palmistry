import os
import cv2
import numpy as np
import json
import time
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from skimage.filters import frangi
import random
import hashlib
import requests
import logging

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & KEYS ---
ASTRO_BASE_URL = "https://api.freeastroapi.com"
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "aa90edcede5379a85560b5db44a773ab0745acd05c734c31a23cdef997e9690e")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY", "AIzaSyA6JLdZhXTV89tLY4z39d2jNvN2iqK4sgI")

if GOOGLE_AI_KEY and GOOGLE_AI_KEY != "YOUR_GOOGLE_AI_KEY_HERE":
    try:
        genai.configure(api_key=GOOGLE_AI_KEY)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        logger.error(f"Gemini Init Error: {e}")
        model = None
else:
    model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
TRAIN_DATA_FILE = 'global_training_data.json'
STATS_FILE = 'stats.json'

# Ensure directories exist (redundant but safe)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- RECOVERY / FALLBACK DATA ---
DEFAULT_KUNDLI = {
    "SUN": "Aries (1st House)", "MOON": "Gemini (3rd House)", 
    "JUPITER": "Leo (5th House)", "MARS": "Scorpio (8th House)",
    "prediction": "The celestial alignment favors your journey with success and clarity."
}

# --- REAL ASTRO API CALLS ---

def get_genuine_horoscope(sign):
    url = f"{ASTRO_BASE_URL}/daily-sign"
    params = {"sign": sign.lower(), "lang": "en"}
    headers = {"x-api-key": ASTRO_API_KEY}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('data', {}).get('prediction', None)
    except Exception as e:
        logger.error(f"Horoscope API Error: {e}")
    return None

def get_genuine_kundli(user_data):
    url = f"{ASTRO_BASE_URL}/full-calculate"
    headers = {"x-api-key": ASTRO_API_KEY, "Content-Type": "application/json"}
    try:
        dob_parts = user_data.get('dob', '1990-01-01').split('-')
        tob_parts = user_data.get('time', '12:00').split(':')
        payload = {
            "year": int(dob_parts[0]), "month": int(dob_parts[1]), "date": int(dob_parts[2]),
            "hours": int(tob_parts[0]), "minutes": int(tob_parts[1]),
            "latitude": user_data.get('lat', 28.61), "longitude": user_data.get('lon', 77.20),
            "timezone": 5.5
        }
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            planets = response.json().get('data', {}).get('planets', {})
            if planets:
                return {
                    "SUN": planets.get('Sun', {}).get('sign', "Aries"),
                    "MOON": planets.get('Moon', {}).get('sign', "Gemini"),
                    "JUPITER": planets.get('Jupiter', {}).get('sign', "Leo"),
                    "MARS": planets.get('Mars', {}).get('sign', "Scorpio")
                }
    except Exception as e:
        logger.error(f"Kundli API Error: {e}")
    return None

# --- API ENDPOINTS ---

@app.route('/')
def status():
    return jsonify({"engine": "AstroAI", "status": "online", "vision": "Pro-Xray 2.0"})

@app.route('/get_horoscope', methods=['GET'])
def get_horoscope():
    sign = request.args.get('sign', 'Aries').capitalize()
    prediction = get_genuine_horoscope(sign)
    
    if not prediction and model:
        try:
            resp = model.generate_content(f"Daily horoscope for {sign} in 2 sentences.")
            prediction = resp.text
        except: pass
    
    if not prediction:
        prediction = f"Your celestial path for {sign} is glowing with potential today."
        
    return jsonify({
        "sign": sign, "prediction": prediction,
        "lucky_number": random.randint(1, 99), "lucky_color": "Gold"
    })

@app.route('/generate_kundli', methods=['POST'])
def generate_kundli():
    data = request.json or {}
    positions = get_genuine_kundli(data) or {
        "SUN": "Aries", "MOON": "Gemini", "JUPITER": "Leo", "MARS": "Scorpio"
    }
    
    prediction = "Celestial alignments indicate a period of positive growth."
    if model:
        try:
            p = f"Vedic Master: Sun in {positions['SUN']}, Moon in {positions['MOON']}. 1-sentence prediction."
            prediction = model.generate_content(p).text
        except: pass

    return jsonify({**positions, "prediction": prediction})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fname = secure_filename(file.filename)
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    
    try:
        img = cv2.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        frangi_img = frangi(denoised, sigmas=range(1, 10, 2), black_ridges=True)
        frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        xray_name = f"xray_{fname}"
        cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), cv2.subtract(210, frangi_norm))
        
        line_count = int(np.sum(frangi_norm > 50) / 800) + 15
        metrics = {"strong_lines": 8, "weak_lines": 12, "depth_index": 8.5, "biometric_id": "ABC123XYZ"}
        
        ai_msg = "Your life matrix suggests a path of high destiny."
        if model:
            try: ai_msg = model.generate_content(f"Palmist: {line_count} lines. 2-sentence divine result.").text
            except: pass

        base_url = "https://palmistry-fk4f.onrender.com"
        return jsonify({
            "line_count": line_count,
            "destiny_score": 95,
            "xray_url": f"{base_url}/static/processed/{xray_name}",
            "metrics": metrics,
            "palm_mapping": {
                "life": {"feature": "Long", "insight": "Great vitality today."},
                "head": {"feature": "Straight", "insight": "Focus on planning."},
                "heart": {"feature": "Stable", "insight": "Emotional harmony."},
                "fate": {"feature": "Clear", "insight": "Career success."},
                "sun": {"feature": "Bright", "insight": "Recognition."},
                "mercury": {"feature": "Good", "insight": "Clear speech."},
                "combined_result": ai_msg
            }
        })
    except Exception as e:
        logger.error(f"Analyze Error: {e}")
        return jsonify({"error": "Internal Processing Error"}), 500

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
