import os
import cv2
import numpy as np
import json
import time
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import random

# --- CONFIGURATION ---
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "aa90edcede5379a85560b5db44a773ab0745acd05c734c31a23cdef997e9690e")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY", "AIzaSyA6JLdZhXTV89tLY4z39d2jNvN2iqK4sgI")

if GOOGLE_AI_KEY:
    genai.configure(api_key=GOOGLE_AI_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_enhanced_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 75, 75)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(denoised, 35, 95)
    combined = cv2.addWeighted(thresh, 0.3, edges, 0.7, 0)
    
    # Create Glowing Cyan X-ray effect
    background = np.full(img.shape, (46, 26, 26), dtype=np.uint8) # Dark Blue #1A1A2E
    mask = combined > 0
    background[mask] = [255, 255, 0] # Cyan [B,G,R]
    
    xray_name = "xray_" + os.path.basename(image_path)
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, xray_name), background)
    
    line_count = int(np.sum(combined > 0) / 1500) + 12
    metrics = {
        "curvature": round(random.uniform(0.7, 0.9), 2),
        "consistency": random.randint(90, 98),
        "pattern": random.choice(["Radial Loop", "Cosmic Arch", "Divine Path"])
    }
    return f"static/processed/{xray_name}", line_count, metrics

@app.route('/')
def status():
    return jsonify({"engine": "AstroAI", "status": "online"})

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    fname = secure_filename(file.filename)
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    
    xray_path, line_count, metrics = process_enhanced_xray(fpath)
    score = 85 + (line_count % 15)
    
    if model:
        prompt = f"Act as a divine palm reader. Analysis shows {line_count} lines. Destiny score is {score}. Provide a mystical prediction in 2 sentences."
        try: ai_msg = model.generate_content(prompt).text
        except: ai_msg = "Your life matrix indicates a surge of creative energy."
    else:
        ai_msg = "Your life matrix indicates a surge of creative energy."

    # Use absolute URL for Render
    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({
        "line_count": line_count,
        "destiny_score": score,
        "xray_url": f"{base_url}/{xray_path}",
        "ai_prediction": ai_msg,
        "metrics": metrics
    })

@app.route('/sync_offline', methods=['POST'])
def sync_offline():
    return jsonify({"status": "success", "synced": len(request.json)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
