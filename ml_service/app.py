"""
FastAPI ML Service for Local Crop Disease Detection
Provides:
- POST /predict: Accepts image upload (jpg/png, max 5MB) + optional crop hint
- GET /classes: Lists all supported crops and their disease classes
- GET /health: Health check and status
- GET /samples/{sample_name}: Serves sample real leaf images for testing
- GET /: Interactive diagnostic dashboard with botanical morphology reporting and mismatch penalties
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from predict import CropDiseasePredictor

# Global predictor instance
predictor: Optional[CropDiseasePredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("[*] Starting up FastAPI ML service. Initializing CropDiseasePredictor...")
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    predictor = CropDiseasePredictor(config_path=config_path)
    print(f"[+] Model loaded. Serving {predictor.num_classes} classes on device: {predictor.device}")
    yield
    print("[*] Shutting down ML service.")

app = FastAPI(
    title="Local Crop Disease Detection ML API",
    description="High-performance edge ML inference service with quality checks, leaf morphology profiling, and uncertainty calibration.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Spring Boot backend and local web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopPrediction(BaseModel):
    label: str
    score: float
    level: Optional[str] = None

class QualityInfo(BaseModel):
    blur_score: float
    brightness: float

class PredictResponse(BaseModel):
    status: str = Field(..., description="ok | low_confidence | poor_image_quality | not_a_leaf | error | species_mismatch | unknown")
    crop: str
    disease: str
    confidence: float
    certainty: str = Field(..., description="high | medium | low")
    level: Optional[str] = Field(None, description="high | medium | low")
    top3: List[TopPrediction]
    # Two-stage confidences (added in v2 - backward-compatible, not consumed by Spring Boot)
    crop_confidence: Optional[float] = Field(None, description="Stage-1: P(Crop) — marginal crop probability")
    disease_confidence: Optional[float] = Field(None, description="Stage-2: P(Disease|Crop) — disease probability conditioned on the identified crop")
    crop_comparison: Optional[List[Dict[str, Any]]] = None
    morphology: Optional[Dict[str, Any]] = None
    mismatch_warning: Optional[str] = None
    quality: QualityInfo
    guidance: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None
    message: str
    raw_top1_class: Optional[str] = None
    raw_top1_crop: Optional[str] = None
    prior_flipped: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    classes_count: int
    uptime_sec: float

start_timestamp = time.time()

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health status and diagnostic info."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "unknown",
        classes_count=predictor.num_classes if predictor else 0,
        uptime_sec=round(time.time() - start_timestamp, 2)
    )

@app.get("/classes")
def get_supported_classes() -> Dict[str, Any]:
    """Returns structured list of supported crops and their corresponding diseases."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")

    crops_map: Dict[str, List[str]] = {}
    for cls_name in predictor.classes:
        parts = cls_name.split("___")
        crop = parts[0]
        disease = parts[1] if len(parts) > 1 else "Unknown"

        if crop not in crops_map:
            crops_map[crop] = []
        crops_map[crop].append(disease)

    return {
        "crops": crops_map,
        "total_classes": predictor.num_classes,
        "classes_list": predictor.classes
    }

SAMPLE_MAP = {
    "bell-pepper-bacterial-spot": "a-bellpepper-leaf.png",
    "wheat-rust": "a-wheat-leaf.png",
    "apple-scab": "apple-fungal-leaf.png",
    "tomato-early-blight": "a-tomato-leaf.png",
    "potato-early-blight": "a-potato-leaf.png",
    "grape-measles": "a-grapes-leaf.png",
    "maize-leaf-spot": "a-maize-leaf.png",
    "cotton-blight": "a-cotton-leaf.png",
    "rice-brown-spot": "a-rice-leaf.png",
    "sugarcane-red-rot": "a-sugarcane-leaf.png",
    # Legacy aliases
    "wheat-yellow-rust": "a-wheat-leaf.png",
    "apple-cedar-rust": "apple-fungal-leaf.png",
    "bell-pepper-healthy": "Bell_Pepper___Healthy_0.jpg"
}

@app.get("/samples/{sample_name}")
def get_sample_leaf(sample_name: str):
    """Serves real photographic sample leaves for 1-click testing."""
    filename = SAMPLE_MAP.get(sample_name)
    if not filename:
        raise HTTPException(status_code=404, detail="Sample not found")

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample_real_leaves")
    filepath = os.path.normpath(os.path.join(sample_dir, filename))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found on disk")

    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    return FileResponse(filepath, media_type=mime, filename=filename)

@app.post("/predict", response_model=PredictResponse)
async def predict_crop_disease(
    image: Optional[UploadFile] = File(None, description="Crop leaf photo in JPG or PNG format (max 5 MB)"),
    file: Optional[UploadFile] = File(None, description="Alternative file upload parameter"),
    crop: Optional[str] = Form(None, description="Optional crop hint to constrain prediction distribution")
):
    """
    Accepts leaf image upload, performs quality screening, morphology extraction,
    calibrated inference, OOD detection, and optional crop conditioning.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML inference engine not initialized.")

    upload_file = image or file
    if upload_file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image file uploaded. Pass 'image' or 'file'.")

    # 1. Validate file extension and MIME type
    filename = (upload_file.filename or "").lower()
    content_type = (upload_file.content_type or "").lower()

    valid_extensions = (".jpg", ".jpeg", ".png")
    valid_mimes = ("image/jpeg", "image/png", "application/octet-stream")

    if not any(filename.endswith(ext) for ext in valid_extensions) and not any(mime in content_type for mime in valid_mimes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{content_type}'. Please upload a valid JPG or PNG image."
        )

    # 2. Read bytes and enforce maximum size constraint (5 MB)
    max_bytes = 5 * 1024 * 1024
    image_bytes = await upload_file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds 5 MB limit ({len(image_bytes)/(1024*1024):.2f} MB)."
        )

    # 3. Execute prediction pipeline
    result = predictor.predict(image_bytes=image_bytes, crop_hint=crop)
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Corrupted or invalid image format. Please upload a valid JPG or PNG image.")
        )
    if "disclaimer" not in result or result["disclaimer"] is None:
        result["disclaimer"] = (
            "Notice: This automated recommendation provides non-binding agricultural decision support. "
            "It is not an agronomist-confirmed clinical diagnosis. Verify physical symptoms in-field before purchasing chemical inputs."
        )
    return result

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the interactive web application for leaf botanical morphology analysis and disease diagnosis."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crop Disease & Health Diagnostic Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #080f0d;
      --card-bg: rgba(16, 28, 23, 0.88);
      --card-border: rgba(52, 211, 153, 0.2);
      --accent-green: #10b981;
      --accent-emerald: #059669;
      --accent-mint: #34d399;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --border: #1a2f26;
      --badge-bg: rgba(16, 185, 129, 0.12);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 15% 15%, #10261f 0%, var(--bg) 65%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 30px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container {
      max-width: 1140px;
      width: 100%;
    }
    header {
      margin-bottom: 25px;
      text-align: center;
    }
    .badge-top {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      background: var(--badge-bg);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: var(--accent-mint);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 12px;
      letter-spacing: 0.3px;
    }
    h1 {
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.6px;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #ffffff 30%, #a7f3d0 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 14.5px;
      max-width: 720px;
      margin: 0 auto;
      line-height: 1.55;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-top: 22px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(14px);
    }
    .card-title {
      font-size: 17px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #ffffff;
    }

    /* Upload & Quick-test samples */
    .dropzone {
      border: 2px dashed rgba(52, 211, 153, 0.38);
      border-radius: 14px;
      padding: 26px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(16, 185, 129, 0.03);
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-mint);
      background: rgba(16, 185, 129, 0.09);
      transform: translateY(-2px);
    }
    .dropzone svg {
      width: 44px;
      height: 44px;
      color: var(--accent-mint);
      margin-bottom: 10px;
    }
    .drop-text {
      font-size: 15px;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 4px;
    }
    .drop-hint {
      font-size: 12px;
      color: var(--text-muted);
    }
    #fileInput { display: none; }
    .preview-wrapper {
      margin-top: 14px;
      display: none;
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      max-height: 220px;
      background: #000;
      border: 1px solid var(--border);
    }
    #imagePreview {
      width: 100%;
      height: 220px;
      object-fit: contain;
      display: block;
    }

    .samples-row {
      margin-top: 14px;
    }
    .samples-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 7px;
    }
    .sample-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .sample-chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(52, 211, 153, 0.25);
      color: #e5e7eb;
      padding: 5px 10px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .sample-chip:hover {
      background: rgba(16, 185, 129, 0.2);
      border-color: var(--accent-mint);
      color: #ffffff;
      transform: translateY(-1px);
    }

    .form-group {
      margin-top: 18px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: #e5e7eb;
      margin-bottom: 6px;
    }
    .select-desc {
      font-size: 11.5px;
      color: var(--text-muted);
      margin-bottom: 8px;
      line-height: 1.4;
    }
    select {
      width: 100%;
      background: #0b1713;
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 11px 14px;
      border-radius: 10px;
      font-size: 14px;
      outline: none;
      cursor: pointer;
      transition: border 0.2s;
    }
    select:focus {
      border-color: var(--accent-green);
    }
    .btn-submit {
      width: 100%;
      margin-top: 20px;
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: #ffffff;
      border: none;
      padding: 13px 20px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      transition: all 0.25s ease;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }
    .btn-submit:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    .btn-submit:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Results Card */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      height: 420px;
      color: var(--text-muted);
      padding: 20px;
    }
    .empty-state svg {
      width: 60px;
      height: 60px;
      margin-bottom: 14px;
      opacity: 0.3;
    }

    /* Mismatch Alert Box */
    .mismatch-alert {
      background: rgba(239, 68, 68, 0.12);
      border: 1.5px solid rgba(239, 68, 68, 0.7);
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 18px;
      display: none;
      animation: fadeIn 0.3s ease-in-out;
    }
    .mismatch-alert-title {
      font-size: 13px;
      font-weight: 800;
      color: #f87171;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }
    .mismatch-alert-body {
      font-size: 12.5px;
      color: #fecaca;
      line-height: 1.5;
    }

    /* Step 1: Leaf Botanical Morphology Card (Top Right) */
    .morphology-box {
      background: rgba(11, 22, 18, 0.9);
      border: 1px solid rgba(52, 211, 153, 0.25);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 18px;
    }
    .morphology-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1px solid rgba(52, 211, 153, 0.15);
      padding-bottom: 8px;
    }
    .morph-step-badge {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      color: var(--accent-mint);
      background: rgba(16, 185, 129, 0.15);
      padding: 3px 8px;
      border-radius: 6px;
    }
    .morph-detected-crop {
      font-size: 16px;
      font-weight: 800;
      color: #ffffff;
    }
    .morph-sci-name {
      font-size: 12px;
      font-style: italic;
      color: var(--text-muted);
    }
    .morph-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 12px;
      font-size: 12px;
    }
    .morph-item {
      background: rgba(255, 255, 255, 0.03);
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .morph-item-label {
      color: var(--text-muted);
      font-size: 10.5px;
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 2px;
    }
    .morph-item-val {
      color: #e5e7eb;
      font-weight: 600;
    }

    /* Step 2: Diagnostic Disease Header */
    .res-header {
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.25);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 18px;
    }
    .res-step-badge {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      color: #38bdf8;
      margin-bottom: 6px;
    }
    .res-crop {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--accent-mint);
      font-weight: 700;
    }
    .res-disease {
      font-size: 22px;
      font-weight: 800;
      color: #ffffff;
      margin-top: 2px;
    }
    .pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .pill {
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .pill-high { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .pill-med { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .pill-low { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .pill-urgency { background: rgba(217, 70, 239, 0.2); color: #e879f9; border: 1px solid rgba(217, 70, 239, 0.4); }

    /* Metric Boxes */
    .metric-row {
      display: flex;
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric-box {
      flex: 1;
      background: rgba(10, 20, 16, 0.7);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      text-align: center;
    }
    .metric-label {
      font-size: 10.5px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }
    .metric-val {
      font-size: 18px;
      font-weight: 800;
      margin-top: 4px;
      color: #f3f4f6;
    }

    /* Step 3: Crop Confidence Comparison */
    .comparison-box {
      background: rgba(10, 20, 16, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 18px;
    }
    .comparison-title {
      font-size: 12.5px;
      font-weight: 700;
      color: #d1d5db;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .crop-bar-row {
      display: flex;
      align-items: center;
      margin-bottom: 9px;
      font-size: 13px;
    }
    .crop-bar-name {
      width: 110px;
      font-weight: 500;
      color: #e5e7eb;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .crop-bar-track {
      flex: 1;
      height: 9px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      margin: 0 10px;
      overflow: hidden;
    }
    .crop-bar-fill-high {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #34d399);
      border-radius: 999px;
      transition: width 0.6s ease;
    }
    .crop-bar-fill-low {
      height: 100%;
      background: rgba(156, 163, 175, 0.35);
      border-radius: 999px;
      transition: width 0.6s ease;
    }
    .crop-bar-pct {
      width: 50px;
      text-align: right;
      font-weight: 600;
      font-size: 12px;
      color: #9ca3af;
    }
    .crop-bar-pct-high {
      color: #34d399;
      font-weight: 700;
    }

    /* Guidance Box */
    .guidance-box {
      background: rgba(13, 26, 21, 0.85);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-top: 16px;
    }
    .guidance-title {
      font-size: 13.5px;
      font-weight: 700;
      color: var(--accent-mint);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .guidance-list {
      list-style-type: none;
      padding-left: 0;
      font-size: 13px;
      color: #d1d5db;
      line-height: 1.5;
    }
    .guidance-list li {
      margin-bottom: 6px;
      position: relative;
      padding-left: 18px;
    }
    .guidance-list li::before {
      content: "•";
      color: var(--accent-green);
      position: absolute;
      left: 4px;
      font-weight: bold;
    }
    .treatment-text {
      font-size: 13px;
      color: #e5e7eb;
      line-height: 1.5;
      background: rgba(16, 185, 129, 0.06);
      border-left: 3px solid var(--accent-green);
      padding: 10px 12px;
      border-radius: 4px;
      margin-top: 8px;
    }
    .disclaimer-box {
      margin-top: 16px;
      font-size: 11px;
      color: #9ca3af;
      line-height: 1.4;
      border-top: 1px solid var(--border);
      padding-top: 12px;
      font-style: italic;
    }

    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.25);
      border-top: 3px solid #ffffff;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    /* Alert Cards */
    .alert-card {
      border-radius: 16px;
      padding: 26px 20px;
      text-align: center;
      margin-top: 15px;
      backdrop-filter: blur(14px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .alert-not-a-leaf { background: rgba(239, 68, 68, 0.12); border: 2px solid rgba(239, 68, 68, 0.75); }
    .alert-blurry { background: rgba(245, 158, 11, 0.12); border: 2px solid rgba(245, 158, 11, 0.75); }
    .alert-icon { font-size: 42px; margin-bottom: 10px; }
    .alert-title-danger { font-size: 24px; font-weight: 800; color: #ef4444; }
    .alert-title-warning { font-size: 22px; font-weight: 800; color: #f59e0b; }
    .alert-subtitle { font-size: 14.5px; font-weight: 600; margin-bottom: 8px; }
    .alert-text { font-size: 13px; color: #9ca3af; line-height: 1.55; max-width: 440px; margin: 0 auto; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-top">
        <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" d="M5 13l4 4L19 7"></path></svg>
        AI-Powered Botanical Morphology & Agronomic Decision Support
      </div>
      <h1>Crop Disease & Health Diagnostic Platform</h1>
      <p class="subtitle">Two-stage foliar pipeline: Automated leaf botanical morphology extraction (geometry, size, shape, margin, color, venation), species recognition, 39-class disease diagnosis, and calibrated crop confidence comparison.</p>
    </header>

    <div class="grid">
      <!-- Input Panel -->
      <div class="card">
        <div class="card-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
          Select Crop Photograph
        </div>
        
        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
          <div class="drop-text" id="dropPrompt">Click to browse or drop leaf photo here</div>
          <div class="drop-hint">Supports JPEG and PNG (up to 5 MB)</div>
          <input type="file" id="fileInput" accept="image/jpeg,image/png">
        </div>

        <!-- Quick-load sample leaves -->
        <div class="samples-row">
          <div class="samples-label">Quick Test Sample Leaves (1-Click):</div>
          <div class="sample-chips">
            <button class="sample-chip" type="button" onclick="loadSample('bell-pepper-bacterial-spot')">🫑 Bell Pepper (Bacterial Spot)</button>
            <button class="sample-chip" type="button" onclick="loadSample('wheat-rust')">🌾 Wheat (Rust Pustules)</button>
            <button class="sample-chip" type="button" onclick="loadSample('apple-scab')">🍎 Apple (Scab)</button>
            <button class="sample-chip" type="button" onclick="loadSample('tomato-early-blight')">🍅 Tomato (Early Blight)</button>
            <button class="sample-chip" type="button" onclick="loadSample('potato-early-blight')">🥔 Potato (Early Blight)</button>
            <button class="sample-chip" type="button" onclick="loadSample('grape-measles')">🍇 Grape (Esca/Measles)</button>
            <button class="sample-chip" type="button" onclick="loadSample('maize-leaf-spot')">🌽 Maize (Leaf Spot)</button>
            <button class="sample-chip" type="button" onclick="loadSample('cotton-blight')">🌱 Cotton (Blight)</button>
            <button class="sample-chip" type="button" onclick="loadSample('rice-brown-spot')">🌾 Rice (Brown Spot)</button>
            <button class="sample-chip" type="button" onclick="loadSample('sugarcane-red-rot')">🎋 Sugarcane (Red Rot)</button>
          </div>
        </div>

        <div class="preview-wrapper" id="previewWrapper">
          <img id="imagePreview" src="" alt="Selected leaf preview">
        </div>

        <div class="form-group">
          <label for="cropSelect">Crop Species (Optional Manual Override)</label>
          <div class="select-desc">Leave on <strong>Auto-Detect</strong> for AI botanical identification. If manually shifted to an incompatible crop, a mismatch penalty is applied.</div>
          <select id="cropSelect">
            <option value="" selected>Auto-Detect Crop (Botanical AI)</option>
            <option value="Bell_Pepper">Bell Pepper</option>
            <option value="Tomato">Tomato</option>
            <option value="Potato">Potato</option>
            <option value="Apple">Apple</option>
            <option value="Maize">Maize (Corn)</option>
            <option value="Wheat">Wheat</option>
            <option value="Cotton">Cotton</option>
            <option value="Sugarcane">Sugarcane</option>
            <option value="Grape">Grape</option>
            <option value="Rice">Rice</option>
          </select>
        </div>

        <button class="btn-submit" id="submitBtn" onclick="runPrediction()" disabled>
          <span class="spinner" id="btnSpinner"></span>
          <span id="btnText">Diagnose Leaf Disease</span>
        </button>
      </div>

      <!-- Result Panel -->
      <div class="card" id="resultsCard">
        <div class="card-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          Diagnostic Report
        </div>

        <div class="empty-state" id="emptyState">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
          <p>Please select or click a sample leaf photo on the left, then click <strong>"Diagnose Leaf Disease"</strong> to see the full botanical report.</p>
        </div>

        <!-- Alert Card: NOT A LEAF -->
        <div class="alert-card alert-not-a-leaf" id="alertNotALeaf" style="display: none;">
          <div class="alert-icon">🚫</div>
          <div class="alert-title-danger">NOT A LEAF</div>
          <div class="alert-subtitle" style="color: #fecaca;">Yeh kisi fasal ki patti nahi hai!</div>
          <div class="alert-text">The uploaded image is not recognized as foliage. Please upload a clear photograph of a crop leaf.</div>
        </div>

        <!-- Alert Card: BLURRY / UNCLEAR -->
        <div class="alert-card alert-blurry" id="alertBlurry" style="display: none;">
          <div class="alert-icon">📷</div>
          <div class="alert-title-warning">PHOTO CLEAR NAHI HAI</div>
          <div class="alert-subtitle" id="blurrySubtitle" style="color: #fef3c7;">Photo bahut dhundhli (blurry) ya andhere mein hai.</div>
          <div class="alert-text" id="blurryDesc">Image sharpness is too low. Kripya camera saaf karke clear photo dubara upload karein.</div>
        </div>

        <div id="resultContent" style="display: none;">
          <!-- Species Mismatch Banner -->
          <div class="mismatch-alert" id="mismatchAlert">
            <div class="mismatch-alert-title">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
              Crop Species Mismatch Warning
            </div>
            <div class="mismatch-alert-body" id="mismatchAlertBody"></div>
          </div>

          <!-- Step 1: Leaf Botanical & Morphological Feature Profile -->
          <div class="morphology-box" id="morphologyBox">
            <div class="morphology-header">
              <div>
                <div class="morph-detected-crop" id="morphDetectedCrop">Bell Pepper</div>
                <div class="morph-sci-name" id="morphSciName">Capsicum annuum • Solanaceae</div>
              </div>
              <div class="morph-step-badge">Step 1: Botanical Morphology</div>
            </div>
            <div class="morph-grid">
              <div class="morph-item">
                <div class="morph-item-label">Leaf Geometry / Shape</div>
                <div class="morph-item-val" id="morphShape">Simple Ovate-Elliptic</div>
              </div>
              <div class="morph-item">
                <div class="morph-item-label">Margin Structure</div>
                <div class="morph-item-val" id="morphMargin">Entire (Smooth)</div>
              </div>
              <div class="morph-item">
                <div class="morph-item-label">Color & Spectrum</div>
                <div class="morph-item-val" id="morphColor">Vibrant Emerald Green</div>
              </div>
              <div class="morph-item">
                <div class="morph-item-label">Venation Pattern</div>
                <div class="morph-item-val" id="morphVenation">Pinnate Reticulate</div>
              </div>
              <div class="morph-item">
                <div class="morph-item-label">Solidity / Circularity</div>
                <div class="morph-item-val" id="morphSolidity">0.86 / 0.48</div>
              </div>
              <div class="morph-item">
                <div class="morph-item-label">Lesion Symptomatology</div>
                <div class="morph-item-val" id="morphLesions">Necrotic spots detected</div>
              </div>
            </div>
          </div>

          <!-- Step 2: Crop & Disease Diagnostic Card -->
          <div class="res-header">
            <div class="res-step-badge">Step 2: Pathology Diagnosis</div>
            <div class="res-crop" id="resCropName">Crop</div>
            <div class="res-disease" id="resDiseaseName">Disease</div>
            <div class="pills">
              <span class="pill" id="certaintyPill">High Certainty</span>
              <span class="pill pill-urgency" id="urgencyPill">Urgency: Medium</span>
              <span class="pill" id="categoryPill">Pathology: Fungal</span>
            </div>
          </div>

          <div class="metric-row">
            <div class="metric-box">
              <div class="metric-label">Calibrated Confidence</div>
              <div class="metric-val" id="resConfidence">0%</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Sharpness (Blur)</div>
              <div class="metric-val" id="resBlur">0</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Brightness</div>
              <div class="metric-val" id="resBrightness">0</div>
            </div>
          </div>

          <!-- Step 3: Crop Confidence Comparison Breakdown -->
          <div class="comparison-box" id="comparisonBox">
            <div class="comparison-title">
              <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
              Crop Confidence Comparison (Calibrated Softmax)
            </div>
            <div id="cropComparisonList"></div>
          </div>

          <!-- Step 4: Agronomic Measures & Treatment -->
          <div class="guidance-box" id="guidanceBox">
            <div class="guidance-title">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
              Preventive Agronomic Measures
            </div>
            <ul class="guidance-list" id="preventiveList">
              <li>Loading preventive guidelines...</li>
            </ul>

            <div class="guidance-title" style="margin-top: 14px;">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
              Treatment & Control Strategy
            </div>
            <div class="treatment-text" id="treatmentText">Loading recommended treatments...</div>
          </div>

          <div class="disclaimer-box" id="disclaimerText">
            Notice: This automated recommendation provides non-binding agricultural decision support.
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePreview = document.getElementById('imagePreview');
    const submitBtn = document.getElementById('submitBtn');
    let selectedFile = null;

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(event => {
      dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(event => {
      dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });
    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    function handleFile(file) {
      if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
        alert('Please upload a JPG or PNG image.');
        return;
      }
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewWrapper.style.display = 'block';
        document.getElementById('dropPrompt').innerText = file.name;
        submitBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    }

    const SAMPLE_CROPS = {
      'bell-pepper-bacterial-spot': 'Bell_Pepper',
      'wheat-rust': 'Wheat',
      'apple-scab': 'Apple',
      'tomato-early-blight': 'Tomato',
      'potato-early-blight': 'Potato',
      'grape-measles': 'Grape',
      'maize-leaf-spot': 'Maize',
      'cotton-blight': 'Cotton',
      'rice-brown-spot': 'Rice',
      'sugarcane-red-rot': 'Sugarcane',
      'wheat-yellow-rust': 'Wheat',
      'apple-cedar-rust': 'Apple',
      'bell-pepper-healthy': 'Bell_Pepper'
    };

    async function loadSample(sampleName) {
      try {
        const res = await fetch('/samples/' + sampleName);
        if (!res.ok) throw new Error('Sample not available');
        const blob = await res.blob();
        const ext = (blob.type === 'image/png') ? '.png' : '.jpg';
        const file = new File([blob], sampleName + ext, { type: blob.type || 'image/jpeg' });
        handleFile(file);
        if (SAMPLE_CROPS[sampleName]) {
          document.getElementById('cropSelect').value = SAMPLE_CROPS[sampleName];
        }
      } catch (err) {
        alert('Could not load sample: ' + err.message);
      }
    }

    async function runPrediction() {
      if (!selectedFile) return;

      const btnSpinner = document.getElementById('btnSpinner');
      const btnText = document.getElementById('btnText');
      const cropHint = document.getElementById('cropSelect').value;

      submitBtn.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnText.innerText = 'Analyzing Leaf Morphology...';

      const formData = new FormData();
      formData.append('image', selectedFile);
      if (cropHint) {
        formData.append('crop', cropHint);
      }

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Prediction failed');
        }

        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.innerText = 'Diagnose Leaf Disease';
      }
    }

    function renderResults(data) {
      document.getElementById('emptyState').style.display = 'none';
      const alertNotALeaf = document.getElementById('alertNotALeaf');
      const alertBlurry = document.getElementById('alertBlurry');
      const resultContent = document.getElementById('resultContent');

      // 1. Rejection: Not A Leaf
      if (data.status === 'not_a_leaf') {
        alertNotALeaf.style.display = 'block';
        alertBlurry.style.display = 'none';
        resultContent.style.display = 'none';
        if (data.message) {
          const alertDesc = alertNotALeaf.querySelector('.alert-text');
          if (alertDesc) alertDesc.innerText = data.message;
        }
        return;
      }

      // 2. Rejection: Poor Image Quality
      if (data.status === 'poor_image_quality') {
        alertBlurry.style.display = 'block';
        alertNotALeaf.style.display = 'none';
        resultContent.style.display = 'none';
        if (data.message) {
          const blurryDesc = document.getElementById('blurryDesc');
          if (blurryDesc) blurryDesc.innerText = data.message;
        }
        return;
      }

      // 3. Valid Leaf Prediction: Show Full Diagnostic Report
      alertNotALeaf.style.display = 'none';
      alertBlurry.style.display = 'none';
      resultContent.style.display = 'block';

      // Check Species Mismatch
      const mismatchAlert = document.getElementById('mismatchAlert');
      if (data.mismatch_warning) {
        mismatchAlert.style.display = 'block';
        document.getElementById('mismatchAlertBody').innerText = data.mismatch_warning;
      } else {
        mismatchAlert.style.display = 'none';
      }

      // Step 1: Render Botanical & Morphological Feature Report (Top Right)
      const morph = data.morphology || {};
      document.getElementById('morphDetectedCrop').innerText = (morph.detected_crop || (data.crop || 'Unknown')).replace(/_/g, ' ');
      const sci = morph.scientific_name ? `${morph.scientific_name} • ${morph.family || 'Plantae'}` : (morph.family || '');
      document.getElementById('morphSciName').innerText = sci;
      document.getElementById('morphShape').innerText = morph.shape || 'Simple Ovate';
      document.getElementById('morphMargin').innerText = morph.margin || 'Entire';
      document.getElementById('morphColor').innerText = morph.color_tone || 'Emerald Green';
      document.getElementById('morphVenation').innerText = morph.venation || 'Pinnate Reticulate';
      document.getElementById('morphSolidity').innerText = `${morph.solidity || '-'} / ${morph.circularity || '-'}`;
      document.getElementById('morphLesions').innerText = morph.lesion_pattern || 'No lesions detected';

      // Step 2: Diagnostic Disease Header
      document.getElementById('resCropName').innerText = (data.crop || 'Unknown').replace(/_/g, ' ');
      document.getElementById('resDiseaseName').innerText = (data.disease || 'Undetected').replace(/_/g, ' ');
      document.getElementById('resConfidence').innerText = (data.confidence * 100).toFixed(1) + '%';
      
      const q = data.quality || {};
      document.getElementById('resBlur').innerText = q.blur_score ? q.blur_score.toFixed(1) : '-';
      document.getElementById('resBrightness').innerText = q.brightness ? q.brightness.toFixed(1) : '-';

      const certPill = document.getElementById('certaintyPill');
      if (data.status === 'species_mismatch') {
        certPill.innerText = 'Low Certainty (Mismatch)';
        certPill.className = 'pill pill-low';
      } else {
        certPill.innerText = (data.certainty || 'Low') + ' Certainty';
        certPill.className = 'pill ' + (data.certainty === 'high' ? 'pill-high' : data.certainty === 'medium' ? 'pill-med' : 'pill-low');
      }

      // Step 3: Render Calibrated Crop Confidence Comparison Breakdown (never 100%)
      const compList = document.getElementById('cropComparisonList');
      compList.innerHTML = '';
      const comparisons = data.crop_comparison || [];
      if (comparisons.length > 0) {
        comparisons.slice(0, 5).forEach((item, idx) => {
          const row = document.createElement('div');
          row.className = 'crop-bar-row';
          const isTop = idx === 0;
          const pct = (item.confidence * 100).toFixed(1);
          const barWidth = Math.max(item.confidence * 100, isTop ? 12 : 3);
          row.innerHTML = `
            <div class="crop-bar-name" style="${isTop ? 'color: #34d399; font-weight:700;' : ''}">${item.crop}</div>
            <div class="crop-bar-track">
              <div class="${isTop ? 'crop-bar-fill-high' : 'crop-bar-fill-low'}" style="width: ${barWidth}%;"></div>
            </div>
            <div class="crop-bar-pct ${isTop ? 'crop-bar-pct-high' : ''}">${pct}%</div>
          `;
          compList.appendChild(row);
        });
      }

      // Step 4: Guidance & Treatment
      const g = data.guidance || {};
      const urgPill = document.getElementById('urgencyPill');
      urgPill.innerText = 'Urgency: ' + (g.urgency || 'Normal');

      const catPill = document.getElementById('categoryPill');
      catPill.innerText = 'Pathology: ' + (g.category || 'Pathology');

      const list = document.getElementById('preventiveList');
      list.innerHTML = '';
      if (g.preventive_measures && g.preventive_measures.length > 0) {
        g.preventive_measures.forEach(p => {
          const li = document.createElement('li');
          li.innerText = p;
          list.appendChild(li);
        });
      } else {
        list.innerHTML = '<li>Inspect canopy regularly and maintain leaf dryness through drip irrigation.</li>';
      }

      document.getElementById('treatmentText').innerText = g.treatment_summary || data.message || 'Consult your local agricultural extension service.';
      document.getElementById('disclaimerText').innerText = data.disclaimer || 'Notice: Decision-support only.';
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    import sys
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, app_dir=app_dir)
