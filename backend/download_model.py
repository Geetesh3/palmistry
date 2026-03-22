import urllib.request
import os

model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print(f"Downloading model from {model_url}...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")
else:
    print("Model file already exists.")
