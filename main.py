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

def get_gabor_filters():
    """Knowledge: Creates a bank of filters to catch lines at all angles."""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 12): # 12 orientations
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filters.append(kern)
    return filters

def process_gabor(img, filters):
    """Knowledge: Applies the filter bank and takes the maximum response."""
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern)
        np.maximum(accum, fimg, accum)
    return accum

def process_perfect_xray(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0, {}
    
    # 1. Extreme Noise Reduction (Non-Local Means)
    # This is better than Bilateral for preserving biometric textures
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 2. Localized Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 3. Gabor Filter Bank (Multi-directional line grabbing)
    filters = get_gabor_filters()
    gabor_output = process_gabor(enhanced, filters)
    
    # 4. Frangi Filter (Tubular structure validation)
    frangi_img = frangi(enhanced, sigmas=range(1, 8, 2), black_ridges=True)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 5. Blend Gabor (Detail) and Frangi (Structure)
    # This results in the clearest lines ever
    combined_lines = cv2.addWeighted(gabor_output, 0.5, frangi_norm, 0.5, 0)
    
    # 6. Hand Masking (Background cleanup)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    
    # 7. Final Ultra-HD Composition
    # Dark Gray skin base
    skin_base = np.full(gray.shape, 200, dtype=np.uint8) 
    # Subtract combined lines to make them blacker than skin
    final_gray = cv2.subtract(skin_base, combined_lines)
    
    # Final Result with perfect black background
    final_hq = np.zeros_like(final_gray)
    final_hq = cv2.bitwise_and(final_gray, final_gray, mask=mask)
    
    xray_name = "perfect_xray_" + os.path.basename(image_path)
    save_path = os.path.join(PROCESSED_FOLDER, xray_name)
    cv2.imwrite(save_path, final_hq)
    
    # 8. High-Precision Metrics
    line_density = float(np.sum(combined_lines > 100) / np.sum(mask > 0))
    line_count = int(line_density * 500) + 10
    
    metrics = {
        "curvature": round(float(np.mean(combined_lines[combined_lines > 0]) / 255.0), 3),
        "density": round(line_density, 4),
        "pattern": "High-Res Neural Ridge"
    }
    
    return f"static/processed/{xray_name}", line_count, metrics

@app.route('/')
def status():
    return jsonify({"engine": "AstroAI Global Training", "vision": "Perfect-Gabor 4.0", "status": "online"})

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files['image']
    fpath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(fpath)
    
    url_path, line_count, metrics = process_perfect_xray(fpath)
    score = 92 + (line_count % 8)
    
    if model:
        prompt = (f"Analyze palm metrics: Density {metrics['density']}, Curvature {metrics['curvature']}. "
                  f"Provide a 2-sentence highly accurate life reading.")
        try: ai_msg = model.generate_content(prompt).text
        except: ai_msg = "Your life matrix indicates deep wisdom and exceptional creative potential."
    else: ai_msg = "Your life matrix indicates deep wisdom and exceptional creative potential."

    return jsonify({
        "line_count": line_count,
        "destiny_score": score,
        "xray_url": f"https://palmistry-fk4f.onrender.com/{url_path}",
        "ai_prediction": ai_msg,
        "metrics": metrics
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
