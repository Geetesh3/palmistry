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

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & CORRECTED API PATHS ---
# Updated to the working v1 versioned URL from your logs
ASTRO_BASE_URL = "https://api.freeastroapi.com/api/v1"
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "aa90edcede5379a85560b5db44a773ab0745acd05c734c31a23cdef997e9690e")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY", "AIzaSyA6JLdZhXTV89tLY4z39d2jNvN2iqK4sgI")

if GOOGLE_AI_KEY and GOOGLE_AI_KEY != "YOUR_GOOGLE_AI_KEY_HERE":
    try:
        genai.configure(api_key=GOOGLE_AI_KEY)
        model = genai.GenerativeModel('gemini-pro')
    except: model = None
else: model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
STATS_FILE = 'stats.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- REAL ASTRO API CALLS (Corrected Paths) ---

def get_genuine_horoscope(sign):
    # FIXED: Using the working path from your logs
    url = f"{ASTRO_BASE_URL}/horoscope/daily/sign"
    params = {"sign": sign.lower(), "lang": "en"}
    headers = {"x-api-key": ASTRO_API_KEY}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Try to find prediction in the new v1 structure
            return data.get('data', {}).get('prediction', None)
    except Exception as e:
        logger.error(f"Horoscope API Error: {e}")
    return None

def get_genuine_kundli(user_data):
    # FIXED: Attempting versioned calculation path
    url = f"{ASTRO_BASE_URL}/birth-chart/vedic/planetary-positions"
    headers = {"x-api-key": ASTRO_API_KEY, "Content-Type": "application/json"}
    try:
        dob = user_data.get('dob', '1990-01-01').split('-')
        tob = user_data.get('time', '12:00').split(':')
        payload = {
            "year": int(dob[0]), "month": int(dob[1]), "date": int(dob[2]),
            "hours": int(tob[0]), "minutes": int(tob[1]),
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

def update_global_training(scans=0, training=0):
    stats = {"total_scans": 1241, "total_training_samples": 4822}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f: stats = json.load(f)
        except: pass
    stats["total_scans"] += scans
    stats["total_training_samples"] += training
    with open(STATS_FILE, 'w') as f: json.dump(stats, f)
    return stats

# --- CORE VISION ENGINE ---
def process_perfect_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    frangi_img = frangi(enhanced, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    skin_base = np.full(gray.shape, 215, dtype=np.uint8)
    final_xray = cv2.subtract(skin_base, frangi_norm)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_hq = cv2.bitwise_and(final_xray, final_xray, mask=mask)
    xray_name = f"perfect_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), final_hq)
    metrics = {"strong_lines": int(np.sum(frangi_norm > 160)/1000)+5, "weak_lines": int(np.sum(frangi_norm > 60)/1000)+10, "depth_index": 8.5, "biometric_id": hashlib.md5(frangi_norm.tobytes()).hexdigest()[:8].upper()}
    return f"static/processed/{xray_name}", int(np.sum(frangi_norm > 50)/850)+12, metrics

# --- API ENDPOINTS ---

@app.route('/')
def status():
    s = update_global_training(0, 0)
    return jsonify({"engine": "AstroAI", "status": "online", "total_ai_trained": s["total_training_samples"], "no_of_scanned_data": s["total_scans"]})

@app.route('/get_horoscope', methods=['GET'])
def get_horoscope():
    sign = request.args.get('sign', 'Aries').capitalize()
    update_global_training(0, 1)
    prediction = get_genuine_horoscope(sign)
    if not prediction and model:
        try: prediction = model.generate_content(f"Daily horoscope for {sign} in 2 sentences.").text
        except: prediction = f"The stars align for {sign} today."
    if not prediction: prediction = f"A day of balance for {sign}."
    return jsonify({"sign": sign, "prediction": prediction, "lucky_number": random.randint(1, 99), "lucky_color": "Gold"})

@app.route('/generate_kundli', methods=['POST'])
def generate_kundli():
    update_global_training(0, 2)
    data = request.json or {}
    positions = get_genuine_kundli(data) or {"SUN": "Aries", "MOON": "Gemini", "JUPITER": "Leo", "MARS": "Scorpio"}
    prediction = "Celestial alignment favors your journey."
    if model:
        try: prediction = model.generate_content(f"Vedic Master: Sun in {positions['SUN']}. 1-sentence prediction.").text
        except: pass
    return jsonify({**positions, "prediction": prediction})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fname = secure_filename(file.filename)
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    url_path, line_count, metrics = process_perfect_xray(fpath)
    update_global_training(1, 5)
    ai_msg = "Your matrix indicates strength."
    if model:
        try: ai_msg = model.generate_content(f"Palmist: {line_count} lines. 2-sentence mystical result.").text
        except: pass
    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({"line_count": line_count, "destiny_score": 95, "xray_url": f"{base_url}/{url_path}", "metrics": metrics, "palm_mapping": {"life": {"feature": "Long", "insight": "Good vitality."}, "head": {"feature": "Straight", "insight": "Logical day."}, "heart": {"feature": "Clear", "insight": "Balance."}, "fate": {"feature": "Strong", "insight": "Success."}, "sun": {"feature": "Visible", "insight": "Recognition."}, "mercury": {"feature": "Active", "insight": "Great talk."}, "combined_result": ai_msg}})

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
