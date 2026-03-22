import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import os
import base64
import httpx
from dotenv import load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv() # Load ASTRO_API_KEY from .env

app = FastAPI()

# Configure CORS for local and production
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://*.onrender.com", # Allows your frontend if hosted on Render
    "https://*.vercel.app",    # Common frontend host
    "https://*.netlify.app",   # Common frontend host
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production ease with multiple frontends, or specify exact URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve absolute path for model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')

# Auto-download model if missing (Crucial for Render/Cloud)
if not os.path.exists(model_path):
    print("Model missing. Downloading celestial weights...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# Load MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# --- External API Config (freeastroapi.com) ---
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY")
ASTRO_BASE_URL = "https://json.freeastroapi.com/v1/"

# --- Knowledge Base ---
ZODIAC_ELEMENTS = {
    "Aries": "Fire", "Leo": "Fire", "Sagittarius": "Fire",
    "Taurus": "Earth", "Virgo": "Earth", "Capricorn": "Earth",
    "Gemini": "Air", "Libra": "Air", "Aquarius": "Air",
    "Cancer": "Water", "Scorpio": "Water", "Pisces": "Water"
}

# --- Models ---
class PalmAnalysis(BaseModel):
    has_hand: bool
    message: str
    lines: dict = {}
    overall_score: int = 0
    personality_trait: str = ""
    line_count: int = 0
    processed_image: str = ""

class KundliRequest(BaseModel):
    name: str
    dob: str
    time: str
    place: str

class CompatibilityRequest(BaseModel):
    sign1: str
    sign2: str

class NumerologyRequest(BaseModel):
    dob: str

class HoroscopeRequest(BaseModel):
    sign: str
    day: str = "today"

# --- Helper for FreeAstroAPI.com ---
async def call_astro_api(endpoint: str, data: dict = None):
    headers = {
        "x-api-key": ASTRO_API_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            url = f"{ASTRO_BASE_URL}{endpoint}"
            # FreeAstroAPI uses POST for most data-driven endpoints
            response = await client.post(url, json=data, headers=headers, timeout=15.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FreeAstroAPI Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"FreeAstroAPI Request failed: {e}")
            return None

import json

# --- Neural Training Cache & Data Store ---
TRAINING_FILE = "training_data.json"

def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r") as f:
            return json.load(f)
    return []

def save_training_data(data):
    current = load_training_data()
    current.append(data)
    with open(TRAINING_FILE, "w") as f:
        json.dump(current, f)

KNOWLEDGE_BASE = {
    "sync_status": "Initial",
    "data_points": len(load_training_data()),
    "last_sync": "Never",
    "refined_readings": {}
}

@app.get("/sync-cosmos")
async def sync_cosmos():
    print("Initializing Cosmic Sync...")
    # Seed sync with a generic request
    real_data = await call_astro_api("horoscope/daily", {"zodiac_sign": "aries"})
    
    if real_data:
        print("Sync successful.")
        KNOWLEDGE_BASE["sync_status"] = "Synchronized"
        KNOWLEDGE_BASE["data_points"] = len(load_training_data()) + 250
        KNOWLEDGE_BASE["last_sync"] = "Just Now"
        return {"status": "success", "data_points": KNOWLEDGE_BASE["data_points"]}
    
    return {"status": "failed", "message": "API method mismatch or key inactive."}

@app.get("/ai-status")
async def get_ai_status():
    KNOWLEDGE_BASE["data_points"] = len(load_training_data()) + (250 if KNOWLEDGE_BASE["sync_status"] == "Synchronized" else 0)
    return KNOWLEDGE_BASE

@app.get("/global-training")
async def get_global_training():
    data = load_training_data()
    return {"total_users_trained": len(data), "recent_readings": data[-5:]}

# --- Expert Logic: Dynamic Precision Palmistry ---
def analyze_palm_expert(image_np, landmarks):
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    edges = cv2.Canny(blurred, 30, 85)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    refined_lines = cv2.bitwise_and(edges, thresh)
    
    indices = [0, 5, 9, 13, 17]
    poly_pts = np.array([[landmarks[i]['x'] * w, landmarks[i]['y'] * h] for i in indices], np.int32)
    mask = np.zeros_like(refined_lines)
    hull = cv2.convexHull(poly_pts)
    cv2.fillConvexPoly(mask, hull, 255)
    
    roi_lines = cv2.bitwise_and(refined_lines, mask)
    line_count = int(np.sum(roi_lines > 0))

    colored_lines = cv2.cvtColor(refined_lines, cv2.COLOR_GRAY2BGR)
    cv2.polylines(colored_lines, [hull], True, (76, 29, 149), 2) 
    colored_lines[roi_lines > 0] = [89, 160, 197] 
    
    _, buffer = cv2.imencode('.jpg', colored_lines)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    palm_h = abs(landmarks[0]['y'] - landmarks[9]['y'])
    palm_w = abs(landmarks[5]['x'] - landmarks[17]['x'])
    ratio = palm_w / palm_h if palm_h != 0 else 1
    trait = "Analytical Architect" if ratio > 0.8 else "Intuitive Visionary"
    
    results = {
        "life_line": {"strength": int(min(99, 45 + (line_count // 350))), "reading": "Immense vitality and resilience detected in your physical matrix."},
        "heart_line": {"strength": int(min(98, 50 + (line_count // 400))), "reading": "Profound emotional depth and empathetic capacity identified."},
        "head_line": {"strength": int(min(97, 55 + (line_count // 380))), "reading": "Highly adaptive cognitive processing and strategic foresight."}
    }
    
    overall = int((results["life_line"]["strength"] + results["heart_line"]["strength"] + results["head_line"]["strength"]) // 3)
    return results, overall, trait, line_count, img_base64

# --- Endpoints ---

@app.post("/horoscope")
async def get_horoscope(req: HoroscopeRequest):
    # FreeAstroAPI mapping
    endpoint = "horoscope/daily"
    data = {"zodiac_sign": req.sign.lower()}
    real_data = await call_astro_api(endpoint, data)
    
    if real_data and "status" in real_data and real_data["status"] == 200:
        p = real_data["data"]["prediction"]
        return {
            "sign": req.sign,
            "daily": p,
            "lucky_number": 7,
            "lucky_color": "Gold",
            "mood": "Harmonious",
            "categories": {"love": "Focus on clarity.", "career": "Growth awaits.", "health": "Steady."}
        }

    # Fallback
    elem = ZODIAC_ELEMENTS.get(req.sign, "Fire")
    return {"sign": req.sign, "daily": "Mars drives ambition today.", "categories": {"love": "Passion peaks.", "career": "Success near.", "health": "Steady."}, "lucky_number": 9, "lucky_color": "Indigo", "mood": "Inspired"}

@app.post("/kundli")
async def generate_kundli(req: KundliRequest):
    # Parse DOB: YYYY-MM-DD
    try:
        y, m, d = map(int, req.dob.split('-'))
        # Parse Time: HH:MM AM/PM
        t_parts = req.time.split()
        h, mn = map(int, t_parts[0].split(':'))
        if len(t_parts) > 1 and t_parts[1].upper() == 'PM' and h < 12: h += 12
        if len(t_parts) > 1 and t_parts[1].upper() == 'AM' and h == 12: h = 0
        
        payload = {
            "year": y, "month": m, "day": d,
            "hour": h, "min": mn,
            "lat": 19.07, "lon": 72.87, "tzone": 5.5
        }
        real_data = await call_astro_api("kundli/birth_details", payload)
        
        if real_data and real_data["status"] == 200:
            d_res = real_data["data"]
            return {
                "ascendant": d_res.get("ascendant", "Leo"),
                "sun_sign": d_res.get("sun_sign", "Aries"),
                "moon_sign": d_res.get("moon_sign", "Taurus"),
                "summary": "Full natal matrix computed from live cosmic data."
            }
    except Exception as e:
        print(f"Kundli parse error: {e}")

    return {"ascendant": "Scorpio", "sun_sign": "Aries", "moon_sign": "Rohini", "summary": "Destiny matrix computed using expert fallback logic."}

@app.post("/compatibility")
async def check_compatibility(req: CompatibilityRequest):
    score = 92 if req.sign1 == req.sign2 else 75
    return {"score": score, "harmony": "High", "reading": "Soul connection identified."}

@app.post("/numerology")
async def calculate_numerology(req: NumerologyRequest):
    lp = sum([int(d) for d in req.dob if d.isdigit()])
    while lp > 9 and lp not in [11, 22, 33]: lp = sum(int(d) for d in str(lp))
    return {"life_path": lp, "reading": "Path of personal mastery."}

@app.get("/moon-phase")
async def get_moon():
    return {"phase": "Waxing Gibbous", "illumination": "82%", "energy": "Manifestation"}

@app.post("/analyze")
async def analyze_palm(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: raise HTTPException(status_code=400, detail="Corrupted Image")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        if not res.hand_landmarks: return PalmAnalysis(has_hand=False, message="No palm identified.")
        landmarks = [{"x": l.x, "y": l.y, "z": l.z} for l in res.hand_landmarks[0]]
        lines, score, trait, count, img = analyze_palm_expert(image, landmarks)
        
        import time
        training_entry = {"timestamp": int(time.time()), "score": score, "line_count": count, "trait": trait}
        save_training_data(training_entry)
        
        return PalmAnalysis(has_hand=True, message="Analysis Complete.", lines=lines, overall_score=score, personality_trait=trait, line_count=count, processed_image=img)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
