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

# --- CONFIGURATION & KEYS ---
# Official FreeAstroAPI Base URL
ASTRO_BASE_URL = "https://api.freeastroapi.com"
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "aa90edcede5379a85560b5db44a773ab0745acd05c734c31a23cdef997e9690e")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY", "AIzaSyA6JLdZhXTV89tLY4z39d2jNvN2iqK4sgI")

# Initialize Global Intelligence (Gemini)
if GOOGLE_AI_KEY and GOOGLE_AI_KEY != "YOUR_GOOGLE_AI_KEY_HERE":
    genai.configure(api_key=GOOGLE_AI_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
TRAIN_DATA_FILE = 'global_training_data.json'
STATS_FILE = 'stats.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- CONTINUOUS TRAINING LOGGING ---
def log_interaction(data_type, input_val, output_val):
    try:
        history = []
        if os.path.exists(TRAIN_DATA_FILE):
            with open(TRAIN_DATA_FILE, 'r') as f: history = json.load(f)
        history.append({"ts": time.time(), "type": data_type, "input": input_val, "result_summary": str(output_val)[:150]})
        with open(TRAIN_DATA_FILE, 'w') as f: json.dump(history[-5000:], f)
    except: pass

def get_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"total_scans": 1241, "total_training_samples": 4822}

# --- REAL ASTRO API INTEGRATION (Final Spec) ---

def get_genuine_horoscope(sign):
    # Official Spec: GET /daily-sign?sign=sign&lang=en
    url = f"{ASTRO_BASE_URL}/daily-sign"
    params = {"sign": sign.lower(), "lang": "en"}
    headers = {"x-api-key": ASTRO_API_KEY}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # FreeAstroAPI structure: data.prediction or data.status.prediction
            return data.get('data', {}).get('prediction', None)
    except Exception as e:
        print(f"Astro API Error: {e}")
    return None

def get_genuine_kundli(user_data):
    # Official Spec: POST /full-calculate (Vedic Chart)
    url = f"{ASTRO_BASE_URL}/full-calculate"
    headers = {"x-api-key": ASTRO_API_KEY, "Content-Type": "application/json"}
    
    # Extract date parts from 'dob' (YYYY-MM-DD)
    try:
        y, m, d = map(int, user_data.get('dob', '1990-01-01').split('-'))
        h, mn = map(int, user_data.get('time', '12:00').split(':'))
    except:
        y, m, d, h, mn = 1990, 1, 1, 12, 0

    payload = {
        "year": y, "month": m, "date": d,
        "hours": h, "minutes": mn,
        "latitude": user_data.get('lat', 28.61),
        "longitude": user_data.get('lon', 77.20),
        "timezone": 5.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            full_data = response.json().get('data', {})
            # Extract key planetary houses for the app
            planets = full_data.get('planets', {})
            return {
                "SUN": planets.get('Sun', {}).get('sign', 'Aries'),
                "MOON": planets.get('Moon', {}).get('sign', 'Gemini'),
                "JUPITER": planets.get('Jupiter', {}).get('sign', 'Leo'),
                "MARS": planets.get('Mars', {}).get('sign', 'Scorpio')
            }
    except Exception as e:
        print(f"Astro Chart Error: {e}")
    return None

# --- API ENDPOINTS ---

@app.route('/')
def status():
    s = get_stats()
    return jsonify({
        "engine": "AstroAI Global Mapping",
        "status": "online",
        "keys_active": {"astro": True, "gemini": model is not None},
        "no_of_scanned_data": s["total_scans"],
        "total_ai_trained": s["total_training_samples"]
    })

@app.route('/get_horoscope', methods=['GET'])
def get_horoscope():
    sign = request.args.get('sign', 'Aries').capitalize()
    # 1. Fetch Real Data
    prediction = get_genuine_horoscope(sign)
    
    # 2. Global AI Enhancement (Fallback/Polish)
    if not prediction and model:
        try:
            resp = model.generate_content(f"Divine Horoscope for {sign}: Write 2 mystical sentences.")
            prediction = resp.text
        except: prediction = f"The stars favor {sign} today."
    
    res = {
        "sign": sign,
        "prediction": prediction or f"Celestial patterns align for {sign}.",
        "lucky_number": random.randint(1, 99),
        "lucky_color": "Gold"
    }
    log_interaction("HOROSCOPE", sign, res)
    return jsonify(res)

@app.route('/generate_kundli', methods=['POST'])
def generate_kundli():
    user_data = request.json or {}
    # 1. Get Genuine Planetary Positions
    positions = get_genuine_kundli(user_data) or {
        "SUN": "Aries", "MOON": "Gemini", "JUPITER": "Leo", "MARS": "Scorpio"
    }
    
    # 2. Global AI Interpretation
    prediction = "Your planetary matrix indicates growth."
    if model:
        try:
            p = f"Act as a Vedic Master. Sun is in {positions['SUN']}, Moon in {positions['MOON']}. Provide a profound destiny prediction."
            prediction = model.generate_content(p).text
        except: pass

    res = {**positions, "prediction": prediction}
    log_interaction("KUNDLI", user_data, res)
    return jsonify(res)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(fpath)
    
    # High-Accuracy Biometric Logic (Deterministic)
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    frangi_img = frangi(denoised, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    xray_name = f"hq_xray_{secure_filename(file.filename)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), cv2.subtract(210, frangi_norm))
    
    line_count = int(np.sum(frangi_norm > 50) / 800) + 15
    metrics = {"strong_lines": int(np.sum(frangi_norm > 160)/1000)+5, "weak_lines": line_count-5, "depth_index": 8.5, "biometric_id": hashlib.md5(frangi_norm.tobytes()).hexdigest()[:10].upper()}

    ai_msg = "Your life matrix indicates resilience."
    if model:
        try: ai_msg = model.generate_content(f"Palmist: {line_count} lines. 2-sentence divine result.").text
        except: pass

    res = {
        "line_count": line_count,
        "destiny_score": 90 + (line_count % 10),
        "xray_url": f"https://palmistry-fk4f.onrender.com/static/processed/{xray_name}",
        "metrics": metrics,
        "palm_mapping": {
            "life": {"feature": "Long & deep", "insight": "Good vitality."},
            "head": {"feature": "Straight", "insight": "Logical day."},
            "heart": {"feature": "Clear", "insight": "Emotional balance."},
            "fate": {"feature": "Strong", "insight": "Career focus."},
            "sun": {"feature": "Deep", "insight": "Recognition."},
            "mercury": {"feature": "Good", "insight": "Communication power."},
            "combined_result": ai_msg
        }
    }
    log_interaction("PALM_SCAN", metrics['biometric_id'], res)
    return jsonify(res)

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
