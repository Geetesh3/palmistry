import os
import cv2
import numpy as np
import json
import time
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from skimage.filters import frangi
from supabase import create_client, Client
import random
import hashlib
import requests

# --- HYPER-CLOUD CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL_HERE")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY_HERE")
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "aa90edcede5379a85560b5db44a773ab0745acd05c734c31a23cdef997e9690e")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY", "AIzaSyA6JLdZhXTV89tLY4z39d2jNvN2iqK4sgI")

# Initialize Cloud DB
supabase: Client = None
if SUPABASE_URL != "YOUR_SUPABASE_URL_HERE":
    try: supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except: pass

# Initialize AI
if GOOGLE_AI_KEY:
    genai.configure(api_key=GOOGLE_AI_KEY)
    model = genai.GenerativeModel('gemini-pro')
else: model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- CLOUD TRAINING SYSTEM ---
def log_to_supabase(data_type, payload):
    """Knowledge: Securely pushes training data to Supabase Cloud DB."""
    if supabase:
        try:
            entry = {
                "type": data_type,
                "data": json.dumps(payload),
                "created_at": "now()"
            }
            supabase.table("global_training").insert(entry).execute()
        except Exception as e:
            print(f"Cloud DB Error: {e}")

# --- VISION PROCESSING (FINAL HQ) ---
def process_biometric_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    frangi_img = frangi(enhanced, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    skin_base = np.full(gray.shape, 215, dtype=np.uint8)
    final_hq = cv2.subtract(skin_base, frangi_norm)
    
    xray_name = f"cloud_hq_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), final_hq)
    
    metrics = {
        "line_count": int(np.sum(frangi_norm > 50) / 800) + 15,
        "depth": round(float(np.mean(frangi_norm[frangi_norm > 50]) / 25.5), 1),
        "bio_id": hashlib.md5(frangi_norm.tobytes()).hexdigest()[:10].upper()
    }
    return f"static/processed/{xray_name}", metrics

# --- API ENDPOINTS ---

@app.route('/')
def status():
    return jsonify({
        "system": "AstroAI Hyper-Cloud",
        "cloud_db": "Supabase" if supabase else "Offline",
        "processor": "Render AI",
        "intelligence": "Gemini-Pro"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(fpath)
    
    url_path, metrics = process_biometric_xray(fpath)
    
    ai_msg = "Your life matrix indicates deep strength."
    if model:
        try: ai_msg = model.generate_content(f"Palm Biometrics: {metrics['line_count']} lines. 2-sentence divine result.").text
        except: pass

    res = {
        "line_count": metrics['line_count'],
        "destiny_score": 92,
        "xray_url": f"https://palmistry-fk4f.onrender.com/{url_path}",
        "metrics": metrics,
        "palm_mapping": {"combined_result": ai_msg}
    }
    
    # Push to Supabase for Global Training
    log_to_supabase("PALM_SCAN", res)
    
    return jsonify(res)

@app.route('/sync_offline', methods=['POST'])
def sync_offline():
    """Receives batch data from Android and pushes to Supabase."""
    data_list = request.json
    if not data_list: return jsonify({"status": "no data"}), 400
    
    for entry in data_list:
        log_to_supabase("OFFLINE_SYNC", entry)
        
    return jsonify({"status": "Cloud Sync Success", "synced": len(data_list)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
