"""
server/api.py
-------------
FastAPI server exposing the Air Writing CNN model via a stateless REST API.

Endpoints:
  GET  /health   — liveness probe (Render health check)
  POST /predict  — character recognition from a base64-encoded JPEG image

The model is loaded once at startup (lifespan event).  Each request is
independent: no session, no trajectory buffer server-side.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolve project root so imports of src.* work identically to the local app
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.predictor import CharacterPredictor  # noqa: E402
from src.renderer import _recenter_centroid  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("air_writing")

# ---------------------------------------------------------------------------
# Model path — override via env var MODEL_PATH
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "model", "model.h5"),
)

_predictor: Optional[CharacterPredictor] = None


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _predictor
    logger.info("Loading model: %s", MODEL_PATH)
    _predictor = CharacterPredictor(model_path=MODEL_PATH)
    logger.info("Model ready — %d classes", _predictor.num_classes)
    yield
    _predictor = None
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App + CORS (allow Flutter web + mobile)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Air Writing API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG image")
    already_rendered: bool = Field(
        False,
        description="If true the image is already a 28×28 grayscale array "
        "ready for the model — skip preprocessing.",
    )


class CharScore(BaseModel):
    char: str
    confidence: float


class PredictResponse(BaseModel):
    char: str
    confidence: float
    top3: List[CharScore]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"service": "air-writing-api", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "num_classes": _predictor.num_classes if _predictor else 0,
    }


def _decode_image(b64: str) -> np.ndarray:
    """Base64 string → grayscale uint8 numpy array."""
    try:
        raw = base64.b64decode(b64)
        buf = np.frombuffer(raw, dtype=np.uint8)
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")
    if gray is None:
        raise HTTPException(status_code=400, detail="Image decoding returned None")
    return gray


def _preprocess(gray: np.ndarray, already_rendered: bool) -> np.ndarray:
    """Convert a decoded grayscale image to a 28×28 float32 [0,1] MNIST-style array."""
    if already_rendered:
        img28 = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        return img28.astype(np.float32) / 255.0

    # MNIST convention: white stroke on black background
    if float(gray.mean()) > 127.0:
        gray = 255 - gray

    _, gray = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(gray)
    if coords is None:
        raise HTTPException(status_code=422, detail="No visible stroke in image")

    # Crop to bounding box of the stroke
    x, y, w, h = cv2.boundingRect(coords)
    roi = gray[y : y + h, x : x + w]

    # Fit into 20×20 box preserving aspect ratio (MNIST centering convention)
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    img28 = canvas.astype(np.float32) / 255.0
    img28 = _recenter_centroid(img28)
    return img28


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    gray = _decode_image(req.image_b64)
    img28 = _preprocess(gray, req.already_rendered)
    label, conf, topk = _predictor.predict_topk(img28, tta=False, k=3)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("'%s' %.1f%%  (%.0f ms)", label, conf * 100, elapsed_ms)

    return PredictResponse(
        char=label,
        confidence=round(conf, 4),
        top3=[CharScore(char=c, confidence=round(p, 4)) for c, p in topk],
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
    )
