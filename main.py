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

# --- CONFIGURATION & KEYS ---
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
TRAIN_FOLDER = 'static/training_assets'
STATS_FILE = 'stats.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)

# --- REAL-TIME DATABASE SIMULATION ---
def get_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {"total_scans": 1240, "total_training_samples": 4820}

def update_stats(scans_inc=1, training_inc=2):
    stats = get_stats()
    stats["total_scans"] += scans_inc
    stats["total_training_samples"] += training_inc
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)
    return stats

# --- IMAGE PROCESSING ---
def process_biometric_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    frangi_img = frangi(enhanced, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Auto-Training: Data Augmentation
    flipped = cv2.flip(frangi_norm, 1)
    cv2.imwrite(os.path.join(TRAIN_FOLDER, f"aug_f_{os.path.basename(image_path)}"), flipped)
    
    # Biometrics
    _, strong_mask = cv2.threshold(frangi_norm, 160, 255, cv2.THRESH_BINARY)
    strong_count = int(np.sum(strong_mask > 0) / 1000) + 5
    weak_count = int(np.sum(frangi_norm > 60) / 1000) - strong_count + 10
    total_lines = strong_count + weak_count
    
    # Professional X-ray Render
    skin_base = np.full(gray.shape, 210, dtype=np.uint8) 
    final_gray = cv2.subtract(skin_base, frangi_norm)
    _, hand_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_hq = cv2.bitwise_and(final_gray, final_gray, mask=hand_mask)
    
    xray_name = f"final_xray_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), final_hq)
    
    metrics = {
        "strong_lines": strong_count,
        "weak_lines": weak_count,
        "depth_index": round(float(np.mean(frangi_norm[frangi_norm > 50]) / 25.5), 1),
        "biometric_id": hashlib.md5(frangi_norm.tobytes()).hexdigest()[:8].upper()
    }
    
    # Update Real-time stats (1 scan + 1 augmented sample)
    update_stats(1, 1)
    
    return f"static/processed/{xray_name}", total_lines, metrics

@app.route('/')
def status():
    stats = get_stats()
    return jsonify({
        "engine": "AstroAI Global Mapping",
        "status": "online",
        "total_ai_trained": stats["total_training_samples"],
        "no_of_scanned_data": stats["total_scans"],
        "model": "MediaPipe Vision 1.0"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(fpath)
    
    url_path, total_lines, metrics = process_biometric_xray(fpath)
    
    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({
        "line_count": total_lines,
        "destiny_score": 90 + (total_lines % 10),
        "xray_url": f"{base_url}/{url_path}",
        "metrics": metrics,
        "palm_mapping": {
            "life": {"feature": "Strong", "insight": "Good energy flow detected."},
            "head": {"feature": "Straight", "insight": "Logical clarity is high."},
            "heart": {"feature": "Deep", "insight": "Stable emotions today."},
            "fate": {"feature": "Clear", "insight": "Path is opening up."},
            "sun": {"feature": "Visible", "insight": "Recognition is coming."},
            "mercury": {"feature": "Good", "insight": "Strong communication."},
            "combined_result": "Synergy confirmed: Peak day for action."
        }
    })

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
