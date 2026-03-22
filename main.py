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

if GOOGLE_AI_KEY and GOOGLE_AI_KEY != "YOUR_GOOGLE_AI_KEY_HERE":
    genai.configure(api_key=GOOGLE_AI_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    model = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
TRAIN_FOLDER = 'static/training_assets'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)

def process_high_quality_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    
    # 1. High-Def Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized detail
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 2. Ridge Detection (Black Top Hat)
    # This highlights the "valleys" (lines) in the palm texture
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Clean and Threshold
    _, thresh = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(thresh, 3)
    
    # 4. Extract main structural lines using Canny on the denoised original
    v = np.median(enhanced_gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(enhanced_gray, lower, upper)
    
    # 5. Combine Ridge details with Structural edges
    combined = cv2.addWeighted(denoised, 0.6, edges, 0.4, 0)
    
    # 6. High-Quality X-ray Rendering (Glow Effect)
    # Create a dark cosmic background
    h, w = img.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (46, 26, 26) # Dark Blue #1A1A2E
    
    # Make lines GLOW Cyan
    # Layer 1: The thin sharp lines
    cyan_lines = np.zeros_like(canvas)
    cyan_lines[combined > 0] = (255, 255, 0) # Cyan [B,G,R]
    
    # Layer 2: A blurred "glow" version of the lines
    glow = cv2.GaussianBlur(cyan_lines, (15, 15), 0)
    
    # Merge layers
    final_xray = cv2.addWeighted(canvas, 1.0, glow, 0.5, 0)
    final_xray = cv2.addWeighted(final_xray, 1.0, cyan_lines, 1.0, 0)
    
    xray_name = "hq_xray_" + os.path.basename(image_path)
    save_path = os.path.join(PROCESSED_FOLDER, xray_name)
    cv2.imwrite(save_path, final_xray)
    
    # 7. Knowledge Extraction
    line_count = int(np.sum(combined > 0) / 1000) + 15
    metrics = {
        "curvature": round(random.uniform(0.75, 0.92), 2),
        "consistency": random.randint(92, 99),
        "pattern": random.choice(["Divine Flow", "Sacred Geometry", "Infinite Arch"])
    }
    
    return f"static/processed/{xray_name}", line_count, metrics

@app.route('/')
def status():
    return jsonify({
        "engine": "AstroAI Global Training",
        "keys_active": {"astro": True, "gemini": model is not None},
        "model": "MediaPipe Vision 1.0",
        "status": "online"
    })

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
    
    # Process with High-Quality Pipeline
    xray_url_path, line_count, metrics = process_high_quality_xray(fpath)
    score = 88 + (line_count % 12)
    
    if model:
        prompt = f"Divine Palm Analysis: {line_count} lines, {metrics['curvature']} curvature. Write a mystical life-line reading and a final destiny outcome in 2 short sentences."
        try: ai_msg = model.generate_content(prompt).text
        except: ai_msg = "Your life matrix suggests a soul of great depth and an unfolding path of prosperity."
    else:
        ai_msg = "Your life matrix suggests a soul of great depth and an unfolding path of prosperity."

    base_url = "https://palmistry-fk4f.onrender.com"
    return jsonify({
        "line_count": line_count,
        "destiny_score": score,
        "xray_url": f"{base_url}/{xray_url_path}",
        "ai_prediction": ai_msg,
        "metrics": metrics
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
