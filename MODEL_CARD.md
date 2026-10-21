# Model Card: Local Crop Disease Detector (MobileNetV3-Large)

## Model Overview
- **Model Architecture**: MobileNetV3-Large with customized 39-class classification head
- **Pretrained Weights**: ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`)
- **Active Model File**: `ml_service/model/model.pt` (Loaded directly by the FastAPI ML service in `predict.py`)
- **Exported ONNX Model**: `ml_service/model/model.onnx` (39-class ONNX graph for high-throughput deployment)
- **Input Resolution**: Configurable `img_size` (default 224, supports 288 or 320; consistent across training, evaluation, calibration, and inference)
- **Model Size**: ~16.8 MB (CPU friendly, sub-second latency for edge/server inference)
- **Framework**: PyTorch 2.4+ / Torchvision / ONNX Runtime
- **Intended Domain**: Agricultural decision support for smallholder farmers and extension officers

---

## Supported Crops and Disease Classes (39 Classes)
All classes follow the strict naming schema `Crop___Disease`. Every supported crop includes a dedicated `Healthy` baseline class.

| Crop | Disease Classes | Healthy Class |
| :--- | :--- | :--- |
| **Tomato** (7) | `Tomato___Bacterial_spot`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Leaf_Mold`, `Tomato___Septoria_leaf_spot`, `Tomato___Yellow_Leaf_Curl_Virus` | `Tomato___Healthy` |
| **Potato** (3) | `Potato___Early_blight`, `Potato___Late_blight` | `Potato___Healthy` |
| **Maize** (4) | `Maize___Common_rust`, `Maize___Gray_leaf_spot`, `Maize___Northern_Leaf_Blight` | `Maize___Healthy` |
| **Bell Pepper** (2) | `Bell_Pepper___Bacterial_spot` | `Bell_Pepper___Healthy` |
| **Grape** (4) | `Grape___Black_rot`, `Grape___Esca_Black_Measles`, `Grape___Leaf_blight` | `Grape___Healthy` |
| **Apple** (4) | `Apple___Apple_scab`, `Apple___Black_rot`, `Apple___Cedar_apple_rust` | `Apple___Healthy` |
| **Rice** (4) | `Rice___Bacterial_leaf_blight`, `Rice___Brown_spot`, `Rice___Leaf_blast` | `Rice___Healthy` |
| **Wheat** (4) | `Wheat___Brown_rust`, `Wheat___Yellow_rust`, `Wheat___Septoria` | `Wheat___Healthy` |
| **Cotton** (4) | `Cotton___Bacterial_blight`, `Cotton___Curl_virus`, `Cotton___Fusarium_wilt` | `Cotton___Healthy` |
| **Sugarcane** (3)| `Sugarcane___Red_rot`, `Sugarcane___Mosaic` | `Sugarcane___Healthy` |

---

## Data Pipeline & Split Strategy

### 1. Source Datasets
- **Laboratory-Controlled**: PlantVillage (Apple, Bell Pepper, Grape, Maize, Potato, Tomato) with uniform gray/black backdrops.
- **In-Field Real-World**: PlantDoc (natural field backgrounds, ambient lighting), Kaggle Rice, Kaggle Wheat, Kaggle Cotton, Kaggle Sugarcane.
- **Source Tracking**: The `source` and `type` (`lab` vs `field`) columns are strictly preserved in all manifests (`manifest.csv`, `train_manifest.csv`, `val_manifest.csv`, `test_lab_manifest.csv`, `test_field_manifest.csv`) for shortcut-learning audit and Leave-One-Source-Out evaluation.

### 2. Leakage Prevention & Deduplication
- **Multi-Orientation Perceptual Hashing (pHash)**: Compares pHash not only on the original image, but also on horizontal flips and 90°, 180°, and 270° rotations with a Hamming distance threshold $\le 4$. This prevents rotated or mirrored copies of the same leaf from leaking across train and test splits. Reports standard duplicates and extra rotated duplicates removed.
- **Corrupt File Pre-Flight & Failure Gate**: `PIL.Image.verify()` pre-flight check validates all manifest paths before training, and `CropDiseaseDataset` enforces `max_failure_rate: 0.005` (0.5%), aborting training if corruption exceeds threshold.

### 3. Training Augmentation Pipeline
- **Rotation Corner Fill**: `RandomRotationWithFill` fills rotated corners using the image's mean color (or reflect-padding), avoiding artificial black border artifacts that never occur at inference.
- **Real Background Replacement**: `FieldBackgroundVariation` segments the leaf using excess-green ($2G - R - B$) and HSV color thresholding (matching `predict.py` logic). With probability $p = 0.4$, it replaces non-leaf pixels with random crops from `data.backgrounds_dir`, or falls back to procedural soil, grass, and sky textures. Applied strictly when leaf coverage is between 10% and 90%, preventing leaf erasure while breaking lab-background shortcut learning.

---

## Evaluation Benchmark & Shortcut-Learning Auditing

### 1. Benchmark Splits & Empirical Measurements
Full training metrics across complete multi-epoch runs on external compute are tracked below:

| Evaluation Benchmark Split | Accuracy | Macro F1-Score | Weighted F1-Score | Expected Calibration Error (ECE) |
| :--- | :--- | :--- | :--- | :--- |
| **Laboratory Test Set (`test_lab`)** | *[To be measured on full run]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |
| **Field Test Set (`test_field`)** | *[To be measured on full run]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |
| **Validation Set (`val`)** | *[To be measured on full run]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |

### 2. Measured Real Leaf Benchmark (`ml_service/eval_real.py`)
Empirical test across 72 real photographic field leaf samples (`data/sample_real_leaves`):
- **Raw Top-1 Crop Accuracy**: 75.00% (54 / 72 samples)
- **Top-1 Crop Accuracy (with clamped morphology priors)**: 75.00% (54 / 72 samples)
- **Prior Flip Rate**: Clamping prior multipliers to $[0.8, 1.25]$ reduced prior flips from 8.33% down to 2.78%, eliminating misprediction flips.
- **Species Mismatch Accuracy with Valid Hint**: 0.0% false mismatch rate when the user provides a correct crop hint.

### 3. Leave-One-Source-Out (LOSO) Cross-Domain Protocol
`04_evaluate.py` provides `--leave-out-source <SOURCE>` and `--loso` modes:
- Trains or evaluates the model by holding out an entire dataset source (e.g. `PlantDoc` or `Kaggle_Rice`).
- Measures out-of-domain transfer accuracy to quantify the true generalization gap on unseen agricultural environments.

### 4. Per-Source & Per-Class Shortcut Risk Audit (`ml_training/source_class_report.csv`)
`04_evaluate.py` audits every class in the manifest, flagging any class where a single dataset source makes up $\ge 90\%$ of images as a **Shortcut-Learning Risk** (`shortcut_risk: True`), warning developers where the model might learn dataset-specific backgrounds rather than plant pathology.

---

## Inference Decision Logic (`ml_service/predict.py`)

### 1. Enforced Image Quality Gate
- Resizes image so the longer dimension is 512px before calculating blur.
- Variance of Laplacian $< 80.0$ rejects with status `poor_image_quality` ("Photo is blurry").
- Grayscale brightness $< 40.0$ (too dark) or $> 225.0$ (overexposed / glare) rejects with status `poor_image_quality`.

### 2. Vegetative Foliar Pre-Filter & Neutral Rejection (`is_leaf_present`)
- Fast OpenCV foliar filter validates presence of chlorophyll ($2G - R - B > 5$) and vegetative hue ranges ($25 \le H \le 95$).
- Restricts necrotic lesions and rust ranges to require spatial adjacency to chlorophyll tissue, preventing human skin, soil, or wood from counting as leaves.
- Flags documents, white paper, and screenshots when neutral pixel ratio exceeds `max_neutral_ratio: 0.70`.
- Minimum leaf contour area ratio must exceed 0.03.

### 3. Free Energy Out-of-Distribution (OOD) Detection
- Primary OOD signal: Free Energy score $E(x) = -T \cdot \log \sum_i \exp(z_i / T)$.
- When $E(x) > \text{energy\_threshold}$ (configurable in `config.yaml`, default $-3.0$), rejects random noise, non-crop photos, and unclassifiable inputs with status `not_a_leaf`.

### 4. Crop Hint Conditioning & Normalization
- Extensively normalizes crop hints (lowercasing, punctuation stripping, and synonym mapping for English, scientific, and regional names: e.g. `corn` $\to$ `Maize`, `paddy` $\to$ `Rice`, `capsicum` $\to$ `Bell_Pepper`, `sugar cane` $\to$ `Sugarcane`).
- If a valid crop hint is provided, it is selected as the effective crop.
- Status `species_mismatch` triggers **only** when the hinted crop's total CNN probability mass (`crop_mass`) is below `mismatch_threshold: 0.10`.

### 5. Grounded Confidence Formula
- Unbiased confidence formula:
  $$\text{Confidence} = P(\text{disease} \mid \text{effective\_crop}) \times \text{crop\_mass}$$
- Artificial $0.65$ floor and $0.94$ cap, smoothing factors ($0.88 / 0.12$), and arbitrary penalties have been completely removed.
- Certainty tiers are fully natural and reachable:
  - **High**: $\ge 0.85$
  - **Medium**: $0.60 \le p < 0.85$
  - **Low**: $< 0.60$ (returns status `low_confidence`)

### 6. Morphology Priors
- **Disabled by default** (`inference.use_morphology_priors: false`).
- When enabled, multipliers are strictly clamped to $[0.8, 1.25]$ to act only as gentle tie-breakers, preventing false crop overrides.

---

## Limitations and Ethical Considerations
- **Decision Support Only**: This system does NOT provide a legally binding or clinical diagnosis. All API responses include an agronomic disclaimer.
- **Single-Leaf Focus**: The model is trained on close-up single-leaf photos; canopy-level drone aerial shots will fail quality or OOD checks.
- **Chemical Treatment Risk**: Farmers are advised never to apply toxic or high-cost chemicals without corroborating field symptoms with an agricultural extension officer.
