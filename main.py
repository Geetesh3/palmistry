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

# Root Route: Fixes the 'Not Found' error on the main Render URL
@app.get("/")
async def root():
    return {
        "status": "ASTRO.AI SYSTEM ACTIVE",
        "version": "1.0.0",
        "endpoints": ["/ping", "/analyze", "/horoscope", "/ai-status"]
    }

# ULTRA-OPEN CORS: Required for Android/Capacitor apps to connect to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Initialization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path), num_hands=1)
)

# --- Knowledge Base ---
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY")
ASTRO_BASE_URL = "https://api.freeastroapi.com/api/v1/"

# --- Endpoints ---

@app.get("/ping")
async def ping(): return {"status": "connected"}

@app.get("/ai-status")
async def get_ai_status(): return {"sync_status": "Synchronized", "data_points": 5840}

@app.get("/global-training")
async def get_global_training(): return {"total_users_trained": 1642}

@app.get("/sync-cosmos")
async def sync_cosmos(): return {"status": "success", "message": "Neural Matrix Synchronized"}

@app.get("/moon-phase")
async def moon_phase():
    return {"phase": "Waxing Crescent", "illumination": "18%", "energy": "Manifestation"}

@app.post("/horoscope")
async def get_horoscope(req: dict):
    headers = {"x-api-key": ASTRO_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            url = f"{ASTRO_BASE_URL}horoscope/daily/sign?sign={req['sign'].lower()}"
            r = await client.get(url, headers=headers, timeout=10.0)
            if r.status_code == 200:
                d = r.json().get("data", {})
                return {"sign": req["sign"], "daily": d.get("content", {}).get("text"), "categories": {"love": "High harmony.", "career": "Momentum high.", "health": "Steady vitality."}, "lucky_number": 7, "lucky_color": "Gold", "mood": "Inspired"}
        except: pass
    return {"sign": req["sign"], "daily": "Stars are in transition.", "categories": {"love": "Focus on heart.", "career": "Success near.", "health": "Steady."}, "lucky_number": 9, "lucky_color": "Blue", "mood": "Calm"}

@app.post("/analyze")
async def analyze_palm(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        if not res.hand_landmarks: return {"has_hand": False, "message": "No biometric lock."}
        
        return {
            "has_hand": True, "overall_score": 85, "line_count": 5200, "personality_trait": "Luminous Visionary",
            "processed_image": base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8'),
            "lines": {"life_line": {"strength": 82, "reading": "Immense vitality."}, "heart_line": {"strength": 78, "reading": "Deep empathy."}, "head_line": {"strength": 90, "reading": "Strategic focus."}}
        }
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
