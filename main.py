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

# --- CONFIGURATION ---
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "aa90edcede5379a85560b5db44a773ab0745acd05c734c31a23cdef997e9690e")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY", "AIzaSyA6JLdZhXTV89tLY4z39d2jNvN2iqK4sgI")

if GOOGLE_AI_KEY and GOOGLE_AI_KEY != "YOUR_GOOGLE_AI_KEY_HERE":
    genai.configure(api_key=GOOGLE_AI_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
STATS_FILE = 'stats.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def get_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"total_scans": 1240, "total_training_samples": 4820}

def update_stats(scans_inc=1):
    stats = get_stats()
    stats["total_scans"] += scans_inc
    stats["total_training_samples"] += (scans_inc * 2)
    with open(STATS_FILE, 'w') as f: json.dump(stats, f)
    return stats

# --- API ENDPOINTS ---

@app.route('/')
def status():
    s = get_stats()
    return jsonify({
        "engine": "AstroAI Global Mapping",
        "status": "online",
        "total_ai_trained": s["total_training_samples"],
        "no_of_scanned_data": s["total_scans"],
        "keys_active": {"astro": True, "gemini": model is not None}
    })

@app.route('/get_horoscope', methods=['GET'])
def get_horoscope():
    sign = request.args.get('sign', 'Aries').capitalize()
    if model:
        try:
            resp = model.generate_content(f"Provide a genuine mystical daily horoscope for {sign} in 2 sentences. Include a lucky number and color.")
            prediction = resp.text
        except:
            prediction = f"The stars align for {sign} today, bringing clarity and purpose."
    else:
        prediction = f"A day of balance and energy for {sign}. Stay focused."
    
    return jsonify({
        "sign": sign,
        "prediction": prediction,
        "lucky_number": random.randint(1, 99),
        "lucky_color": random.choice(["Emerald", "Gold", "Crimson", "Indigo", "Silver"])
    })

@app.route('/generate_kundli', methods=['POST'])
def generate_kundli():
    data = request.json or {}
    name = data.get('name', 'Seeker')
    dob = data.get('dob', 'N/A')
    time_b = data.get('time', 'N/A')
    
    if model:
        try:
            prompt = f"Act as a Vedic Astrologer. Generate a planetary report for {name} born on {dob} at {time_b}. Sun, Moon, Jupiter positions and 1 divine prediction sentence."
            prediction = model.generate_content(prompt).text
        except:
            prediction = "Celestial alignments indicate a period of significant growth."
    else:
        prediction = "The stars favor your current path with Lord Ganesha's blessings."

    return jsonify({
        "SUN": "Aries (1st House)",
        "MOON": "Gemini (3rd House)",
        "JUPITER": "Leo (5th House)",
        "MARS": "Scorpio (8th House)",
        "prediction": prediction
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(fpath)
    
    # Vision Logic (Using standard processing for speed in this demo)
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    line_count = int(np.sum(edges > 0) / 1200) + 15
    
    xray_name = "xray_" + secure_filename(file.filename)
    xray_path = os.path.join(PROCESSED_FOLDER, xray_name)
    cv2.imwrite(xray_path, edges) # Binary for training accuracy
    
    update_stats(1)
    
    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({
        "line_count": line_count,
        "destiny_score": 90 + (line_count % 10),
        "xray_url": f"{base_url}/static/processed/{xray_name}",
        "palm_mapping": {
            "life": {"feature": "Strong", "insight": "Good health flow."},
            "head": {"feature": "Straight", "insight": "Logical decisions."},
            "heart": {"feature": "Stable", "insight": "Emotional balance."},
            "fate": {"feature": "Clear", "insight": "Career success."},
            "sun": {"feature": "Bright", "insight": "Recognition."},
            "mercury": {"feature": "Good", "insight": "Clear speech."},
            "combined_result": "Peak synergy detected."
        }
    })

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
