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

# --- CONFIGURATION ---
ASTRO_BASE_URL = "https://api.freeastroapi.com"
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

def process_hq_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    frangi_img = frangi(enhanced, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    strong_lines = int(np.sum(frangi_norm > 160) / 1000) + 5
    weak_lines = int(np.sum(frangi_norm > 60) / 1000) + 10
    line_depth = round(float(np.mean(frangi_norm[frangi_norm > 50]) / 25.5), 1)
    bio_id = hashlib.md5(frangi_norm.tobytes()).hexdigest()[:10].upper()
    
    skin_base = np.full(gray.shape, 210, dtype=np.uint8)
    final_xray = cv2.subtract(skin_base, frangi_norm)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_hq = cv2.bitwise_and(final_gray, final_gray, mask=mask) if 'final_gray' in locals() else final_xray
    final_hq = cv2.bitwise_and(final_hq, final_hq, mask=mask)
    
    xray_name = f"hq_xray_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), final_hq)
    
    metrics = {"strong_lines": strong_lines, "weak_lines": weak_lines, "depth_index": line_depth, "biometric_id": bio_id}
    return f"static/processed/{xray_name}", strong_lines + weak_lines, metrics

@app.route('/')
def health(): return jsonify({"status": "AstroAI Online"})

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
    
    url_path, line_count, metrics = process_hq_xray(fpath)
    score = 90 + (line_count % 10)
    
    ai_msg = "Your life matrix indicates a surge of energy."
    if model:
        try:
            p = f"Act as a professional palm reader. Analysis: {line_count} lines, {metrics['depth_index']} depth. 2-sentence mystical reading."
            ai_msg = model.generate_content(p).text
        except: pass

    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({
        "line_count": line_count,
        "destiny_score": score,
        "xray_url": f"{base_url}/{url_path}",
        "metrics": metrics,
        "palm_mapping": {
            "life": {"feature": "Long & deep", "insight": "Strong energy today."},
            "head": {"feature": "Straight", "insight": "Good planning day."},
            "heart": {"feature": "Clear", "insight": "Emotional harmony."},
            "fate": {"feature": "Strong", "insight": "Career success."},
            "sun": {"feature": "Visible", "insight": "Recognition coming."},
            "mercury": {"feature": "Active", "insight": "Clear communication."},
            "combined_result": ai_msg
        }
    })

@app.route('/get_horoscope', methods=['GET'])
def get_horoscope():
    sign = request.args.get('sign', 'Aries').capitalize()
    return jsonify({"sign": sign, "prediction": f"The stars align for {sign} today.", "lucky_number": 7, "lucky_color": "Gold"})

@app.route('/generate_kundli', methods=['POST'])
def generate_kundli():
    return jsonify({"SUN": "Aries (1st House)", "MOON": "Gemini (3rd House)", "prediction": "Celestial alignments indicate growth."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
