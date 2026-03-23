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

# --- CONFIGURATION ---
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_perfect_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    
    # High-HD Processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 1. Ridge Mapping
    frangi_img = frangi(enhanced, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. X-ray Aesthetic (Dark Lines on Gray Base)
    skin_base = np.full(gray.shape, 215, dtype=np.uint8)
    final_xray = cv2.subtract(skin_base, frangi_norm)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_hq = cv2.bitwise_and(final_xray, final_xray, mask=mask)
    
    xray_name = f"perfect_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), final_hq)
    
    line_count = int(np.sum(frangi_norm > 50) / 850) + 12
    metrics = {
        "strong_lines": int(np.sum(frangi_norm > 160) / 1000) + 5,
        "weak_lines": int(np.sum(frangi_norm > 60) / 1000) + 10,
        "depth_index": round(float(np.mean(frangi_norm[frangi_norm > 40]) / 25.5), 1),
        "biometric_id": hashlib.md5(frangi_norm.tobytes()).hexdigest()[:8].upper()
    }
    return f"static/processed/{xray_name}", line_count, metrics

@app.route('/')
def status():
    return jsonify({"engine": "AstroAI Perfect Mapping", "status": "online"})

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fname = secure_filename(file.filename)
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    
    url_path, line_count, metrics = process_perfect_xray(fpath)
    score = 90 + (line_count % 10)
    
    # 1. Divine AI Result (Gemini)
    ai_msg = "Your life matrix indicates a rare alignment of strength and prosperity."
    if model:
        try:
            p = f"Act as a Master Palmist. Hand has {line_count} lines, Depth Index {metrics['depth_index']}. Provide a 2-sentence highly accurate life result."
            ai_msg = model.generate_content(p).text
        except: pass

    # 2. Complete Mapping
    mapping = {
        "life": {"feature": "Long & Deep", "insight": "Vitality is at its peak. Today is great for physical activities."},
        "head": {"feature": "Straight", "insight": "Focus on logic. A good day for financial or career planning."},
        "heart": {"feature": "Clear", "insight": "Emotional stability is high. Relationships will be harmonious."},
        "fate": {"feature": "Strong", "insight": "Career path is opening. Take the opportunity arriving today."},
        "sun": {"feature": "Visible", "insight": "Success is coming. Your hard work is being recognized by the universe."},
        "mercury": {"feature": "Active", "insight": "Great day for communication and meetings. Speak your truth."},
        "combined_result": ai_msg
    }

    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({
        "line_count": line_count,
        "destiny_score": score,
        "xray_url": f"{base_url}/{url_path}",
        "metrics": metrics,
        "palm_mapping": mapping
    })

@app.route('/generate_kundli', methods=['POST'])
def generate_kundli():
    return jsonify({
        "SUN": "Aries (1st House)", "MOON": "Gemini (3rd House)", 
        "JUPITER": "Leo (5th House)", "MARS": "Scorpio (8th House)",
        "prediction": "The planetary alignment indicates a surge of success and Lord Ganesha's blessings."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
