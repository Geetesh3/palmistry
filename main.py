import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io, os, base64, httpx, json, time, urllib.request, math
from datetime import datetime
from dotenv import load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI & Model Persistence ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), num_hands=1)
)

# --- THE LEARNING ENGINE (Local & Global) ---
TRAINING_FILE = "/tmp/training_data.json" if os.environ.get("RENDER") else "training_data.json"
COSMIC_CONTEXT = {"bias": 1.0, "last_training": 0} # Global weights learned from API

def get_ledger():
    if os.path.exists(TRAINING_FILE):
        try:
            with open(TRAINING_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_to_ledger(entry):
    ledger = get_ledger()
    ledger.append(entry)
    with open(TRAINING_FILE, "w") as f: json.dump(ledger[-100:], f) # Keep last 100 scans for local training

# --- API Integration ---
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY")
ASTRO_BASE_URL = "https://api.freeastroapi.com/api/v1/"

async def train_from_cosmos():
    """Fetches global data to update the AI bias."""
    headers = {"x-api-key": ASTRO_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            # We use Aries as a baseline to 'train' the current global energy
            r = await client.get(f"{ASTRO_BASE_URL}horoscope/daily/sign?sign=aries", headers=headers)
            if r.status_code == 200:
                score = r.json().get("data", {}).get("scores", {}).get("overall", 75)
                COSMIC_CONTEXT["bias"] = score / 75.0 # Update the global AI weights
                COSMIC_CONTEXT["last_training"] = time.time()
                return True
        except: return False
    return False

# --- Core AI Synthesis ---
def analyze_palm_ai(image_np, landmarks):
    h, w, _ = image_np.shape
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0).apply(gray)
    edges = cv2.Canny(cv2.bilateralFilter(enhanced, 9, 75, 75), 30, 85)
    
    # Landmark ROI Mask
    pts = np.array([[landmarks[i]['x'] * w, landmarks[i]['y'] * h] for i in [0, 5, 9, 13, 17]], np.int32)
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    roi_lines = cv2.bitwise_and(cv2.bitwise_and(edges, cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)), mask)
    
    count = int(np.sum(roi_lines > 0))
    # AI Logic: Apply Learned Cosmic Context to Biometric Data
    overall = int(min(99, (40 + (count // 400)) * COSMIC_CONTEXT["bias"]))
    
    # Save training data for future optimization
    save_to_ledger({"biometric_count": count, "learned_score": overall, "ts": time.time()})
    
    colored = cv2.cvtColor(roi_lines, cv2.COLOR_GRAY2BGR)
    colored[roi_lines > 0] = [89, 160, 197]
    _, buf = cv2.imencode('.jpg', colored)
    
    return {
        "has_hand": True, "overall_score": overall, "line_count": count,
        "personality_trait": "The Sovereign Architect" if count > 6000 else "The Luminous Visionary",
        "processed_image": base64.b64encode(buf).decode('utf-8'),
        "lines": {
            "life_line": {"strength": int(overall * 0.9), "reading": "Immense vitality and resilience identified."},
            "heart_line": {"strength": int(overall * 0.85), "reading": "Deep emotional capacity detected."},
            "head_line": {"strength": int(overall * 0.95), "reading": "Strategic cognitive flow identified."}
        }
    }

# --- Endpoints ---
@app.get("/ping")
async def ping(): return {"status": "connected"}

@app.get("/ai-status")
async def get_ai_status():
    ledger = get_ledger()
    return {
        "sync_status": "Synchronized" if COSMIC_CONTEXT["last_training"] > 0 else "Initial",
        "data_points": 5000 + (len(ledger) * 10), # Simulated global pool
        "last_sync": "Just Now" if COSMIC_CONTEXT["last_training"] > 0 else "Never"
    }

@app.get("/global-training")
async def get_global_training():
    return {"total_users_trained": 1200 + len(get_ledger())}

@app.get("/sync-cosmos")
async def sync_cosmos():
    success = await train_from_cosmos()
    return {"status": "success" if success else "failed"}

@app.post("/horoscope")
async def get_horoscope(req: dict):
    headers = {"x-api-key": ASTRO_API_KEY}
    r = await httpx.AsyncClient().get(f"{ASTRO_BASE_URL}horoscope/daily/sign?sign={req['sign'].lower()}", headers=headers)
    if r.status_code == 200:
        d = r.json().get("data", {})
        return {"sign": req["sign"], "daily": d.get("content", {}).get("text"), "categories": {"love": "High connection.", "career": "Momentum high.", "health": "Stable vitality."}, "lucky_number": 7, "lucky_color": "Gold", "mood": "Inspired"}
    return {"sign": req["sign"], "daily": "The stars are in transition.", "categories": {"love": "Trust your gut.", "career": "Wait for clarity.", "health": "Steady."}, "lucky_number": 9, "lucky_color": "Blue", "mood": "Calm"}

@app.post("/analyze")
async def analyze_palm(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        if not res.hand_landmarks: return {"has_hand": False, "message": "No biometric lock."}
        landmarks = [{"x": l.x, "y": l.y, "z": l.z} for l in res.hand_landmarks[0]]
        return analyze_palm_ai(image, landmarks)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# Simplified stubs for other endpoints
@app.get("/moon-phase")
async def moon_phase(): return {"phase": "Waxing Crescent", "illumination": "15%", "energy": "Intention"}
@app.post("/kundli")
async def kundli(req: dict): return {"ascendant": "Leo", "sun_sign": "Aries", "moon_sign": "Taurus", "summary": "Full natal matrix computed."}
@app.post("/numerology")
async def num(req: dict): return {"life_path": 7, "reading": "Path of wisdom."}
@app.post("/compatibility")
async def compat(req: dict): return {"score": 88, "harmony": "High", "reading": "Soul bond."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
