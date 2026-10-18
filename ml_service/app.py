"""
FastAPI ML Service for Local Crop Disease Detection
Provides:
- POST /predict: Accepts image upload (jpg/png, max 5MB) + optional crop hint
- GET /classes: Lists all supported crops and their disease classes
- GET /health: Health check and status
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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
    description="High-performance edge ML inference service with quality checks and uncertainty calibration.",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.responses import HTMLResponse

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

class QualityInfo(BaseModel):
    blur_score: float
    brightness: float

class PredictResponse(BaseModel):
    status: str = Field(..., description="ok | low_confidence | poor_image_quality | not_a_leaf | error")
    crop: str
    disease: str
    confidence: float
    certainty: str = Field(..., description="high | medium | low")
    top3: List[TopPrediction]
    crop_comparison: Optional[List[Dict[str, Any]]] = None
    quality: QualityInfo
    guidance: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None
    message: str

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
    """
    Returns structured list of supported crops and their corresponding diseases
    for dynamic frontend/backend dropdown generation.
    """
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

@app.post("/predict", response_model=PredictResponse)
async def predict_crop_disease(
    image: UploadFile = File(..., description="Crop leaf photo in JPG or PNG format (max 5 MB)"),
    crop: Optional[str] = Form(None, description="Optional crop hint to constrain prediction distribution")
):
    """
    Accepts leaf image upload, performs quality screening, calibrated inference,
    OOD detection, and optional crop conditioning.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML inference engine not initialized.")

    # 1. Validate file extension and MIME type
    filename = (image.filename or "").lower()
    content_type = (image.content_type or "").lower()

    valid_extensions = (".jpg", ".jpeg", ".png")
    valid_mimes = ("image/jpeg", "image/png", "application/octet-stream")

    if not any(filename.endswith(ext) for ext in valid_extensions) and not any(mime in content_type for mime in valid_mimes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{content_type}'. Please upload a valid JPG or PNG image."
        )

    # 2. Read bytes and enforce maximum size constraint (5 MB)
    max_bytes = 5 * 1024 * 1024
    image_bytes = await image.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds 5 MB limit ({len(image_bytes)/(1024*1024):.2f} MB)."
        )

    # 3. Execute prediction pipeline
    result = predictor.predict(image_bytes=image_bytes, crop_hint=crop)
    if "disclaimer" not in result or result["disclaimer"] is None:
        result["disclaimer"] = (
            "Notice: This automated recommendation provides non-binding agricultural decision support. "
            "It is not an agronomist-confirmed clinical diagnosis. Verify physical symptoms in-field before purchasing chemical inputs."
        )
    return result

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves an interactive web application for direct leaf image upload and disease diagnosis."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crop Disease & Health Diagnostic Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1311;
      --card-bg: rgba(18, 30, 26, 0.85);
      --card-border: rgba(52, 211, 153, 0.18);
      --accent-green: #10b981;
      --accent-emerald: #059669;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --border: #1f332c;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: radial-gradient(circle at 15% 20%, #122820 0%, var(--bg) 60%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 30px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container {
      max-width: 1100px;
      width: 100%;
    }
    header {
      margin-bottom: 30px;
      text-align: center;
    }
    .badge-top {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 12px;
    }
    h1 {
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #ffffff 40%, #a7f3d0 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 15px;
      max-width: 650px;
      margin: 0 auto;
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 25px;
      margin-top: 25px;
    }
    @media (max-width: 850px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
      backdrop-filter: blur(12px);
    }
    .card-title {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    /* Upload Box */
    .dropzone {
      border: 2px dashed rgba(52, 211, 153, 0.35);
      border-radius: 14px;
      padding: 28px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(16, 185, 129, 0.03);
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-green);
      background: rgba(16, 185, 129, 0.08);
      transform: translateY(-2px);
    }
    .dropzone svg {
      width: 48px;
      height: 48px;
      color: #34d399;
      margin-bottom: 12px;
    }
    .drop-text {
      font-size: 15px;
      font-weight: 500;
      color: var(--text-main);
      margin-bottom: 6px;
    }
    .drop-hint {
      font-size: 12px;
      color: var(--text-muted);
    }
    #fileInput { display: none; }
    .preview-wrapper {
      margin-top: 16px;
      display: none;
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      max-height: 230px;
      background: #000;
    }
    #imagePreview {
      width: 100%;
      height: 230px;
      object-fit: contain;
      display: block;
    }
    .form-group {
      margin-top: 20px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 500;
      color: #d1d5db;
      margin-bottom: 7px;
    }
    select {
      width: 100%;
      background: #0d1a15;
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
      margin-top: 22px;
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: #ffffff;
      border: none;
      padding: 13px 20px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 600;
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
      opacity: 0.55;
      cursor: not-allowed;
    }
    /* Results Pane */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      height: 380px;
      color: var(--text-muted);
      padding: 20px;
    }
    .empty-state svg {
      width: 60px;
      height: 60px;
      margin-bottom: 15px;
      opacity: 0.35;
    }
    .res-header {
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.25);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 18px;
    }
    .res-crop {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #34d399;
      font-weight: 700;
    }
    .res-disease {
      font-size: 22px;
      font-weight: 700;
      color: #ffffff;
      margin-top: 4px;
    }
    .pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .pill {
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .pill-high { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .pill-med { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .pill-low { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .pill-urgency { background: rgba(217, 70, 239, 0.2); color: #e879f9; border: 1px solid rgba(217, 70, 239, 0.4); }
    
    .metric-row {
      display: flex;
      gap: 15px;
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
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .metric-val {
      font-size: 18px;
      font-weight: 700;
      margin-top: 4px;
      color: #f3f4f6;
    }

    /* Guidance Section */
    .guidance-box {
      background: rgba(13, 26, 21, 0.85);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-top: 16px;
    }
    .guidance-title {
      font-size: 14px;
      font-weight: 600;
      color: #34d399;
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
      background: rgba(16, 185, 129, 0.05);
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

    /* Alert Cards for Not A Leaf & Blurry Photo */
    .alert-card {
      border-radius: 16px;
      padding: 30px 24px;
      text-align: center;
      margin-top: 15px;
      backdrop-filter: blur(14px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .alert-not-a-leaf {
      background: rgba(239, 68, 68, 0.12);
      border: 2px solid rgba(239, 68, 68, 0.75);
    }
    .alert-blurry {
      background: rgba(245, 158, 11, 0.12);
      border: 2px solid rgba(245, 158, 11, 0.75);
    }
    .alert-icon { font-size: 48px; margin-bottom: 12px; }
    .alert-title-danger {
      font-size: 28px;
      font-weight: 800;
      color: #ef4444;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 8px;
    }
    .alert-title-warning {
      font-size: 26px;
      font-weight: 800;
      color: #f59e0b;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 8px;
    }
    .alert-subtitle {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 10px;
    }
    .alert-subtitle-danger { color: #fecaca; }
    .alert-subtitle-warning { color: #fef3c7; }
    .alert-text {
      font-size: 13.5px;
      color: #9ca3af;
      line-height: 1.6;
      max-width: 440px;
      margin: 0 auto;
    }

    /* Crop Comparison Bar Chart */
    .comparison-box {
      background: rgba(10, 20, 16, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-top: 16px;
    }
    .comparison-title {
      font-size: 13px;
      font-weight: 600;
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
      width: 105px;
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
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-top">
        <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        AI-Powered Agronomic Decision Support
      </div>
      <h1>Crop Disease & Health Diagnostic Platform</h1>
      <p class="subtitle">Upload high-resolution foliar photographs for automated leaf quality assessment, 39-class disease identification, and actionable agronomic treatments.</p>
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

        <div class="preview-wrapper" id="previewWrapper">
          <img id="imagePreview" src="" alt="Selected leaf preview">
        </div>

        <div class="form-group">
          <label for="cropSelect">Crop Species (Optional Hint)</label>
          <select id="cropSelect">
            <option value="">Auto-Detect (Search all 10 Crops)</option>
            <option value="Rice">Rice</option>
            <option value="Potato">Potato</option>
            <option value="Tomato">Tomato</option>
            <option value="Apple">Apple</option>
            <option value="Maize">Maize (Corn)</option>
            <option value="Wheat">Wheat</option>
            <option value="Cotton">Cotton</option>
            <option value="Sugarcane">Sugarcane</option>
            <option value="Grape">Grape</option>
            <option value="Bell_Pepper">Bell Pepper</option>
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
          <p>Please select an image on the left and click <strong>"Diagnose Leaf Disease"</strong> to see the full prediction report.</p>
        </div>

        <!-- Alert Card: NOT A LEAF -->
        <div class="alert-card alert-not-a-leaf" id="alertNotALeaf" style="display: none;">
          <div class="alert-icon">🚫</div>
          <div class="alert-title-danger">NOT A LEAF</div>
          <div class="alert-subtitle alert-subtitle-danger">Yeh kisi fasal ki patti nahi hai!</div>
          <div class="alert-text">The uploaded image is not recognized as a plant or crop leaf. Please upload a clear photo of a crop leaf.</div>
        </div>

        <!-- Alert Card: BLURRY / UNCLEAR -->
        <div class="alert-card alert-blurry" id="alertBlurry" style="display: none;">
          <div class="alert-icon">📷</div>
          <div class="alert-title-warning">PHOTO CLEAR NAHI HAI</div>
          <div class="alert-subtitle alert-subtitle-warning" id="blurrySubtitle">Photo bahut dhundhli (blurry) ya andhere mein hai.</div>
          <div class="alert-text" id="blurryDesc">Image sharpness is too low. Kripya camera saaf karke clear photo dubara upload karein.</div>
        </div>

        <div id="resultContent" style="display: none;">
          <div class="res-header">
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
              <div class="metric-label">Confidence</div>
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

          <!-- Crop Confidence Comparison Breakdown -->
          <div class="comparison-box" id="comparisonBox">
            <div class="comparison-title">
              <svg width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
              Crop Confidence Comparison
            </div>
            <div id="cropComparisonList"></div>
          </div>

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
            <div class="treatment-text" id="treatmentText">Loading recommended fungicides and control treatments...</div>
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

    async function runPrediction() {
      if (!selectedFile) return;

      const btnSpinner = document.getElementById('btnSpinner');
      const btnText = document.getElementById('btnText');
      const cropHint = document.getElementById('cropSelect').value;

      submitBtn.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnText.innerText = 'Analyzing Leaf...';

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
        return;
      }

      // 2. Rejection: Poor Image Quality (Blurry or too dark)
      if (data.status === 'poor_image_quality') {
        alertBlurry.style.display = 'block';
        alertNotALeaf.style.display = 'none';
        resultContent.style.display = 'none';
        if (data.message) {
          document.getElementById('blurryDesc').innerText = data.message;
        }
        return;
      }

      // 3. Valid Leaf Prediction: Show Diagnostic Report
      alertNotALeaf.style.display = 'none';
      alertBlurry.style.display = 'none';
      resultContent.style.display = 'block';

      document.getElementById('resCropName').innerText = (data.crop || 'Unknown').replace(/_/g, ' ');
      document.getElementById('resDiseaseName').innerText = (data.disease || 'Undetected').replace(/_/g, ' ');
      document.getElementById('resConfidence').innerText = (data.confidence * 100).toFixed(1) + '%';
      
      const q = data.quality || {};
      document.getElementById('resBlur').innerText = q.blur_score ? q.blur_score.toFixed(1) : '-';
      document.getElementById('resBrightness').innerText = q.brightness ? q.brightness.toFixed(1) : '-';

      const certPill = document.getElementById('certaintyPill');
      certPill.innerText = (data.certainty || 'Low') + ' Certainty';
      certPill.className = 'pill ' + (data.certainty === 'high' ? 'pill-high' : data.certainty === 'medium' ? 'pill-med' : 'pill-low');

      // Render Crop Confidence Comparison Breakdown (High confidence for top crop, low for others)
      const compList = document.getElementById('cropComparisonList');
      compList.innerHTML = '';
      const comparisons = data.crop_comparison || [];
      if (comparisons.length > 0) {
        comparisons.slice(0, 5).forEach((item, idx) => {
          const row = document.createElement('div');
          row.className = 'crop-bar-row';
          const isTop = idx === 0;
          const pct = (item.confidence * 100).toFixed(1);
          const barWidth = Math.max(item.confidence * 100, isTop ? 10 : 3);
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

      const g = data.guidance || {};
      const urgPill = document.getElementById('urgencyPill');
      urgPill.innerText = 'Urgency: ' + (g.urgency || 'Normal');

      const catPill = document.getElementById('categoryPill');
      catPill.innerText = 'Category: ' + (g.category || 'Pathology');

      const list = document.getElementById('preventiveList');
      list.innerHTML = '';
      if (g.preventive_measures && g.preventive_measures.length > 0) {
        g.preventive_measures.forEach(p => {
          const li = document.createElement('li');
          li.innerText = p;
          list.appendChild(li);
        });
      } else {
        list.innerHTML = '<li>Scout plant rows regularly and avoid wet canopy foliage.</li>';
      }

      document.getElementById('treatmentText').innerText = g.treatment_summary || data.message || 'Consult your local extension agronomist.';
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
