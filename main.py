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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_biometric_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 1. Advanced Ridge Extraction
    frangi_img = frangi(enhanced, sigmas=range(1, 10, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Differentiate Strong vs Weak Lines
    # Strong: High probability Frangi response
    # Weak: Lower probability Frangi response
    _, strong_mask = cv2.threshold(frangi_norm, 160, 255, cv2.THRESH_BINARY)
    _, weak_mask = cv2.threshold(frangi_norm, 60, 160, cv2.THRESH_BINARY)
    
    # 3. Calculate Biometrics
    strong_count = int(np.sum(strong_mask > 0) / 1000) + 5
    weak_count = int(np.sum(weak_mask > 0) / 1000) + 10
    total_lines = strong_count + weak_count
    
    line_depth = round(float(np.mean(frangi_norm[frangi_norm > 50]) / 25.5), 1) # 1-10 scale
    consistency = random.randint(92, 99)
    
    # Generate Unique Biometric ID for this hand
    bio_id = hashlib.md5(frangi_norm.tobytes()).hexdigest()[:10].upper()
    
    # 4. Professional Rendering
    skin_base = np.full(gray.shape, 210, dtype=np.uint8) 
    final_gray = cv2.subtract(skin_base, frangi_norm)
    _, hand_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_hq = cv2.bitwise_and(final_gray, final_gray, mask=hand_mask)
    
    xray_name = f"bio_xray_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), final_hq)
    
    metrics = {
        "strong_lines": strong_count,
        "weak_lines": weak_count,
        "depth_index": line_depth,
        "consistency": consistency,
        "biometric_id": bio_id,
        "curvature": round(random.uniform(0.7, 0.9), 2)
    }
    
    return f"static/processed/{xray_name}", total_lines, metrics

def get_palm_mapping(metrics):
    # Mapping based on biometric data
    is_energetic = metrics['strong_lines'] > 10
    is_logical = metrics['curvature'] > 0.8
    
    mapping = {
        "life": {"feature": "Strong" if is_energetic else "Calm", "insight": "High vitality detected. Take initiative today." if is_energetic else "Take time for self-care and rest."},
        "head": {"feature": "Deep" if is_logical else "Creative", "insight": "Perfect day for logical planning." if is_logical else "Trust your intuition and creative flow."},
        "heart": {"feature": "Stable", "insight": "Emotional harmony is favored by your current matrix."},
        "fate": {"feature": "Developing", "insight": "Stay focused on your core goals for maximum growth."},
        "sun": {"feature": "Visible", "insight": "Recognition for your efforts is on the horizon."},
        "mercury": {"feature": "Clear", "insight": "Effective communication will open new doors."},
        "combined_result": "Biometric synergy indicates a day of high productivity and clarity."
    }
    return mapping

@app.route('/')
def status():
    return jsonify({"engine": "AstroAI Biometric Mapping", "status": "online"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(fpath)
    
    url_path, total_lines, metrics = process_biometric_xray(fpath)
    mapping = get_palm_mapping(metrics)
    
    if model:
        try:
            p = f"Act as a Master Palmist. Biometrics: {total_lines} lines, {metrics['depth_index']} depth. Provide a 2-sentence divine daily summary."
            mapping['combined_result'] = model.generate_content(p).text
        except: pass

    return jsonify({
        "line_count": total_lines,
        "destiny_score": 90 + (total_lines % 10),
        "xray_url": f"https://palmistry-fk4f.onrender.com/{url_path}",
        "metrics": metrics,
        "palm_mapping": mapping
    })

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
