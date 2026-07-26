"""FastAPI backend for metro platform crowd density detection.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import detector

app = FastAPI(title="Metro crowd density detection")

# Open during development. Restrict to your own origin before any real use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models": detector.available_models()}


@app.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    model: str = Form("finetuned"),
    confidence: float = Form(0.35),
):
    """Accept one image (or one video frame) and return counts and boxes."""
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(status_code=400, detail="confidence must be between 0.05 and 0.95")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="could not read image")

    array = np.array(image)[:, :, ::-1]  # RGB to BGR for the model

    try:
        return detector.detect(array, model_name=model, confidence=confidence)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"detection failed: {exc}")
