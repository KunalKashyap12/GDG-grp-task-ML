# Local Crop Disease Detector & Agricultural Decision Support System

An end-to-end, production-grade Machine Learning and Spring Boot platform for local crop disease detection. A farmer uploads a leaf photo; the system verifies image quality (blur and illumination), runs calibrated neural inference using **MobileNetV3-Large** across 10 crops and 39 classes, detects out-of-distribution non-leaf photos, conditions predictions on optional crop hints, and attaches actionable preventive agricultural guidance with non-binding decision-support disclaimers.

---

## System Architecture

```mermaid
graph TD
    Client["Farmer / Web Frontend / Mobile App"]
    
    subgraph SpringBoot["Spring Boot 3 Backend (:8080)"]
        PredictCtrl["PredictController (/api/predict, /api/crops)"]
        MlClient["MlClientService (WebClient + Timeout + 1 Retry)"]
        GuidanceSvc["GuidanceService (guidance.json cache)"]
        GlobalEx["GlobalExceptionHandler (Clean JSON Errors)"]
    end

    subgraph FastAPIService["FastAPI ML Service (:8000)"]
        FastAPIApp["app.py (POST /predict, GET /classes, GET /health)"]
        QualityCheck["OpenCV Quality Filter (Laplacian Blur & Brightness)"]
        MobileNet["MobileNetV3-Large Backbone (< 1s CPU Inference)"]
        TempScale["Temperature Scaling (T=1.25, ECE Reduction)"]
        EnergyOOD["Free Energy Out-of-Distribution Detector"]
        CropFilter["Crop Hint Renormalization"]
    end

    Client -->|"POST /api/predict (Multipart: image, optional crop)"| PredictCtrl
    PredictCtrl -->|"Validate MIME & Size (<=5MB)"| PredictCtrl
    PredictCtrl -->|"Forward Image"| MlClient
    MlClient -->|"POST /predict"| FastAPIApp
    FastAPIApp --> QualityCheck
    QualityCheck -->|"Pass"| MobileNet
    MobileNet --> TempScale
    TempScale --> EnergyOOD
    EnergyOOD --> CropFilter
    CropFilter -->|"Normalized Probabilities"| FastAPIApp
    FastAPIApp -->|"ML Result JSON"| MlClient
    MlClient --> PredictCtrl
    PredictCtrl -->|"Lookup Disease Guidance"| GuidanceSvc
    GuidanceSvc -->|"Preventive Advice"| PredictCtrl
    PredictCtrl -->|"Final Response + Agricultural Disclaimer"| Client
```

---

## Directory Structure

```text
gdg-grp-task(ml)/
├── ml_training/                     # Complete PyTorch Training Pipeline
│   ├── config.yaml                  # Hyperparameters, paths, and uncertainty thresholds
│   ├── mapping.yaml                 # Maps raw folder names across datasets to Crop___Disease
│   ├── 01_merge_datasets.py         # Merges datasets, caps at 1500/class, writes manifest.csv
│   ├── 02_filter_and_split.py       # pHash deduplication, corrupt filter, train/val/test splits
│   ├── 03_train.py                  # Two-stage MobileNetV3 transfer learning with AMP & AdamW
│   ├── 04_evaluate.py               # Lab vs field evaluation, confusion matrix, weakest classes
│   ├── 05_calibrate.py              # Temperature scaling calibration, reliability diagram, ECE
│   ├── colab_run.py                 # Free Google Colab T4 GPU automated execution script
│   └── utils/
│       ├── __init__.py
│       ├── quality.py               # Variance of Laplacian blur and brightness assessment
│       ├── augmentations.py         # FieldBackgroundVariation & color/lighting augmentations
│       ├── dataset.py               # PyTorch Dataset and smoothed inverse class weights
│       └── metrics.py               # ECE calculation, source breakdown, weakest class finder
├── ml_service/                      # Production FastAPI ML Inference Microservice
│   ├── app.py                       # FastAPI application (POST /predict, GET /classes, GET /health)
│   ├── predict.py                   # CPU-optimized predictor, quality check, crop conditioning
│   ├── config.yaml                  # Runtime service configuration
│   ├── requirements.txt             # Pinned Python dependencies
│   ├── Dockerfile                   # Container definition for ML inference service
│   └── model/
│       ├── model.pt                 # Trained MobileNetV3 PyTorch checkpoint
│       ├── labels.json              # Class index mapping (39 classes)
│       └── calibration.json         # Learned temperature parameter T and energy threshold
├── backend/                         # Java Spring Boot 3 Decision Support Backend
│   ├── pom.xml                      # Maven configuration with Java 17 and Spring Boot 3.3.4
│   ├── Dockerfile                   # Multi-stage container build
│   └── src/
│       ├── main/
│       │   ├── java/com/cropprotect/backend/
│       │   │   ├── BackendApplication.java
│       │   │   ├── config/
│       │   │   │   ├── WebClientConfig.java      # WebClient timeouts & 16MB buffer
│       │   │   │   └── CorsConfig.java           # Cross-origin request mapping
│       │   │   ├── controller/
│       │   │   │   └── PredictController.java    # POST /api/predict & GET /api/crops
│       │   │   ├── dto/
│       │   │   │   ├── TopPrediction.java
│       │   │   │   ├── QualityInfo.java
│       │   │   │   ├── MlPredictionResponse.java
│       │   │   │   ├── GuidanceInfo.java
│       │   │   │   ├── FinalPredictionResponse.java
│       │   │   │   ├── CropsCatalogResponse.java
│       │   │   │   └── ErrorResponse.java
│       │   │   ├── exception/
│       │   │   │   ├── GlobalExceptionHandler.java
│       │   │   │   ├── MlServiceException.java
│       │   │   │   └── InvalidImageException.java
│       │   │   └── service/
│       │   │       ├── GuidanceService.java      # Loads and indexes guidance.json
│       │   │       └── MlClientService.java      # WebClient integration + 1 retry
│       │   └── resources/
│       │       ├── application.yml               # Server port, multipart limits, ML URL
│       │       └── guidance.json                 # Agronomic guidance for all 39 classes
│       └── test/
│           └── java/com/cropprotect/backend/
│               └── PredictControllerTest.java    # MockMvc unit & integration test suite
├── docker-compose.yml               # Orchestrates ml-service and spring-backend
├── test_client.py                   # Sends 20 unseen images covering edge cases to the API
├── MODEL_CARD.md                    # Model specifications, metrics, and limitations
└── README.md                        # Documentation and run instructions
```

---

## Supported Crops and Standardized Classes (39 Classes)

Every class strictly follows the `Crop___Disease` naming convention across training, labels, the ML service, and the Spring Boot guidance catalog:

1. **Apple** (4): `Apple___Apple_scab`, `Apple___Black_rot`, `Apple___Cedar_apple_rust`, `Apple___Healthy`
2. **Bell Pepper** (2): `Bell_Pepper___Bacterial_spot`, `Bell_Pepper___Healthy`
3. **Cotton** (4): `Cotton___Bacterial_blight`, `Cotton___Curl_virus`, `Cotton___Fusarium_wilt`, `Cotton___Healthy`
4. **Grape** (4): `Grape___Black_rot`, `Grape___Esca_Black_Measles`, `Grape___Leaf_blight`, `Grape___Healthy`
5. **Maize** (4): `Maize___Common_rust`, `Maize___Gray_leaf_spot`, `Maize___Northern_Leaf_Blight`, `Maize___Healthy`
6. **Potato** (3): `Potato___Early_blight`, `Potato___Late_blight`, `Potato___Healthy`
7. **Rice** (4): `Rice___Bacterial_leaf_blight`, `Rice___Brown_spot`, `Rice___Leaf_blast`, `Rice___Healthy`
8. **Sugarcane** (3): `Sugarcane___Mosaic`, `Sugarcane___Red_rot`, `Sugarcane___Healthy`
9. **Tomato** (7): `Tomato___Bacterial_spot`, `Tomato___Early_blight`, `Tomato___Healthy`, `Tomato___Late_blight`, `Tomato___Leaf_Mold`, `Tomato___Septoria_leaf_spot`, `Tomato___Yellow_Leaf_Curl_Virus`
10. **Wheat** (4): `Wheat___Brown_rust`, `Wheat___Healthy`, `Wheat___Septoria`, `Wheat___Yellow_rust`

---

## Execution Guide

### Option 1: Run with Docker Compose (Recommended)

Runs both the FastAPI ML microservice and the Spring Boot backend inside isolated containers:

```bash
# Build and start both containers in detached mode
docker-compose up --build -d

# Verify health status of services
docker-compose ps

# Follow runtime logs
docker-compose logs -f
```

- Spring Boot Backend: `http://localhost:8080`
- FastAPI ML Service: `http://localhost:8000`
- ML Health Check: `http://localhost:8000/health`
- Supported Crops Catalog: `http://localhost:8080/api/crops`

---

### Option 2: Run Services Standalone Locally

#### 1. Start the FastAPI ML Service
```bash
# Navigate to the ML service directory
cd ml_service

# Install dependencies
pip install -r requirements.txt

# Start the Uvicorn server
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### 2. Start the Spring Boot Backend
In a separate terminal:
```bash
cd backend

# Build and run with Maven
./mvnw spring-boot:run
# (Or on Windows cmd: mvnw.cmd spring-boot:run)
```

The backend starts on port `8080` and communicates with the ML service on port `8000`.

---

### Option 3: Training the ML Model on Google Colab (Free T4 GPU)

1. Open a new notebook on [Google Colab](https://colab.research.google.com).
2. Set Runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Clone your repository:
   ```bash
   !git clone https://github.com/<your-username>/crop-disease-detector.git
   %cd crop-disease-detector
   ```
4. Place your raw datasets under `data/raw/` (or run synthetic generation):
   ```bash
   python ml_training/01_merge_datasets.py --generate-synthetic
   ```
5. Run the complete automated training and calibration script:
   ```bash
   python ml_training/colab_run.py
   ```
6. The script generates `model_bundle.zip` containing `model.pt`, `labels.json`, and `calibration.json`. Unpack into `ml_service/model/`.

---

## Running the 20 Unseen Test Images Benchmark

The `test_client.py` script generates and submits 20 unseen images covering normal leaves, blurry images, extreme lighting, out-of-distribution noise, and lowercase crop hints:

```bash
# Test through the Spring Boot Backend
python test_client.py --url http://localhost:8080/api/predict

# Or test directly against the ML microservice
python test_client.py --direct-ml
```

### Sample Benchmark Output Table:
```text
#   | Scenario                  | Crop Hint  | Status           | Predicted Class    | Conf     | Certain | Guidance  | Latency 
-----------------------------------------------------------------------------------------------------------------------------
1   | Tomato leaf with spots    | Tomato     | ok               | Tomato___Early_b.. | 0.94     | high    | Yes       | 28.4ms
2   | Potato leaf with spots    | Potato     | ok               | Potato___Early_b.. | 0.88     | high    | Yes       | 24.1ms
3   | Rice leaf lesion          | Rice       | ok               | Rice___Leaf_blast  | 0.78     | medium  | Yes       | 23.9ms
4   | Wheat rust pustules       | Wheat      | ok               | Wheat___Brown_rust | 0.86     | high    | Yes       | 22.8ms
14  | Out-of-focus camera       | Tomato     | poor_image_quali | None               | 0.00     | low     | No        | 12.3ms
15  | Night/underexposed captur | Rice       | poor_image_quali | None               | 0.00     | low     | No        | 11.8ms
17  | Random texture noise      | -          | not_a_leaf       | None               | 0.03     | low     | No        | 21.2ms
19  | Lowercase crop hint check | cotton     | ok               | Cotton___Healthy   | 0.91     | high    | Yes       | 23.5ms
-----------------------------------------------------------------------------------------------------------------------------
```

---

## API Specifications

### 1. `POST /api/predict` (Spring Boot Gateway)
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` *(Required, File)*: Leaf photograph (`.jpg`, `.jpeg`, `.png`, max 5 MB).
  - `crop` *(Optional, Form Text)*: Crop hint (e.g. `"Tomato"`, `"Rice"`, `"Cotton"`).

#### Success Response (`200 OK`):
```json
{
  "status": "ok",
  "crop": "Tomato",
  "disease": "Early_blight",
  "confidence": 0.9421,
  "certainty": "high",
  "top3": [
    { "label": "Tomato___Early_blight", "score": 0.9421 },
    { "label": "Tomato___Late_blight", "score": 0.0382 },
    { "label": "Tomato___Septoria_leaf_spot", "score": 0.0112 }
  ],
  "quality": {
    "blur_score": 194.52,
    "brightness": 128.4
  },
  "message": "High confidence prediction.",
  "guidance": {
    "disease_name": "Tomato Early Blight (Alternaria linariae)",
    "category": "Fungal",
    "urgency": "Medium",
    "preventive_measures": [
      "Prune bottom 12 inches of foliage once plants are established to prevent soil splash.",
      "Apply 2-3 inches of clean straw or plastic mulch under tomato vines.",
      "Rotate nightshade crops on a minimum 3-year schedule."
    ],
    "treatment_summary": "Apply biofungicide (Bacillus subtilis) or chemical protectors (chlorothalonil, azoxystrobin) at first concentric ring spot."
  },
  "disclaimer": "Notice: This automated recommendation provides non-binding agricultural decision support. It is not an agronomist-confirmed clinical diagnosis. Verify physical symptoms in-field before purchasing chemical inputs.",
  "timestamp": "2026-10-17T07:30:00Z"
}
```

#### Blurry Image Quality Response (`200 OK`):
```json
{
  "status": "poor_image_quality",
  "crop": "unknown",
  "disease": "unknown",
  "confidence": 0.0,
  "certainty": "low",
  "top3": [],
  "quality": {
    "blur_score": 42.1,
    "brightness": 115.3
  },
  "message": "Image is blurry (sharpness score 42.1 < 100.0). Hold camera steady.",
  "guidance": null,
  "disclaimer": "Notice: This automated recommendation provides non-binding agricultural decision support...",
  "timestamp": "2026-10-17T07:30:05Z"
}
```

#### Non-Leaf / Out-of-Distribution Image Response (`200 OK`):
```json
{
  "status": "not_a_leaf",
  "crop": "unknown",
  "disease": "unknown",
  "confidence": 0.026,
  "certainty": "low",
  "top3": [],
  "quality": {
    "blur_score": 1420.5,
    "brightness": 130.1
  },
  "message": "Input image does not exhibit features characteristic of a recognizable crop leaf.",
  "guidance": null,
  "disclaimer": "Notice: This automated recommendation provides non-binding agricultural decision support...",
  "timestamp": "2026-10-17T07:30:10Z"
}
```

---

### 2. `GET /api/crops` (Spring Boot Gateway)
Returns the dynamic catalog of supported crops and disease classes:
```json
{
  "crops": {
    "Tomato": [
      "Bacterial_spot", "Early_blight", "Healthy", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "Yellow_Leaf_Curl_Virus"
    ],
    "Potato": [
      "Early_blight", "Late_blight", "Healthy"
    ],
    "Rice": [
      "Bacterial_leaf_blight", "Brown_spot", "Leaf_blast", "Healthy"
    ]
  },
  "total_classes": 39,
  "classes_list": [
    "Apple___Apple_scab",
    "Tomato___Early_blight",
    "Wheat___Brown_rust"
  ]
}
```

---

## Automated JUnit Integration Testing

The Spring Boot test suite validates prediction parsing, guidance attachment, error translation, and invalid upload handling with MockMvc and Mockito:

```bash
cd backend
mvn test
```

Tests include:
- `testSuccessfulPredictionWithGuidance`: Verifies HTTP 200, status `ok`, certainty `high`, and non-null guidance and disclaimer.
- `testPoorQualityImagePrediction`: Verifies blurry uploads return status `poor_image_quality` without attaching guidance.
- `testInvalidFileType`: Verifies non-image uploads (e.g. PDF) are rejected with HTTP 400 `INVALID_IMAGE_PAYLOAD`.
- `testMlServiceDowntimeErrorHandling`: Verifies connection timeouts or service crashes are mapped to HTTP 503 `ML_SERVICE_UNAVAILABLE`.
- `testGetCropsCatalog`: Verifies dynamic dropdown crop catalog proxying.
