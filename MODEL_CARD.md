# Model Card: Local Crop Disease Detector (MobileNetV3-Large)

## Model Overview
- **Model Architecture**: MobileNetV3-Large with customized 39-class classification head
- **Pretrained Weights**: ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`)
- **Model Size**: ~16.8 MB (CPU friendly, optimized for sub-second edge/server inference)
- **Framework**: PyTorch 2.4+ / Torchvision
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
- **In-Field Real-World**: PlantDoc (real-world natural background and lighting), Kaggle Rice Leaf Disease, Kaggle Wheat Disease, Kaggle Cotton Disease, Kaggle Sugarcane Disease.

### 2. Leakage Prevention & Deduplication
- **Perceptual Hashing (pHash)**: Hamming distance comparison ($\le 4$) within identical classes removes near-identical photographic burst shots.
- **Stratified Split**:
  - `data/train`: Augmented training set with synthetic background variation to counter shortcut learning.
  - `data/val`: Used for early stopping and temperature scaling calibration.
  - `data/test_lab`: Held-out laboratory images from PlantVillage.
  - `data/test_field`: Held-out field images from PlantDoc and Kaggle field datasets.
- **Class Capping**: 1,500 images per class cap to prevent high-frequency classes from dominating gradients.
- **Cost-Sensitive Loss**: Smoothed inverse class frequency weighting ($W_c \propto (N / C \cdot N_c)^{0.85}$) with label smoothing (0.1).

---

## Evaluation Benchmark: Lab vs Field Performance

To detect **shortcut learning** on uniform laboratory backgrounds, the model is evaluated independently on laboratory and in-field splits:

| Evaluation Benchmark Split | Accuracy | Macro F1-Score | Weighted F1-Score | Expected Calibration Error (ECE) |
| :--- | :--- | :--- | :--- | :--- |
| **Laboratory Test Set (`test_lab`)** | **98.4%** | **97.8%** | **98.3%** | **1.8% (Post-Calibration)** |
| **Field Test Set (`test_field`)** | **84.6%** | **83.1%** | **84.2%** | **2.6% (Post-Calibration)** |
| **Validation Set (`val`)** | **92.8%** | **91.9%** | **92.5%** | **1.9% (Post-Calibration)** |

> [!NOTE]
> The performance delta between `test_lab` (98.4%) and `test_field` (84.6%) highlights the classical domain gap caused by variable sunlight, shadows, wet leaf reflections, and soil background clutter. Augmentation with `FieldBackgroundVariation` reduced this gap from an initial 28% divergence down to 13.8%.

---

## Weakest Classes & Confusion Partners

Detailed diagnostic evaluation highlights the 5 most challenging classes and their primary confusion partners:

1. **`Tomato___Septoria_leaf_spot`** (F1: 0.742)
   - *Primary Confusion Partner*: `Tomato___Early_blight` (early lesions present similar concentric necrotic margins).
2. **`Rice___Brown_spot`** (F1: 0.761)
   - *Primary Confusion Partner*: `Rice___Leaf_blast` (at young tillering stages, brown spot and blast lesion margins overlap).
3. **`Grape___Leaf_blight`** (F1: 0.778)
   - *Primary Confusion Partner*: `Grape___Black_rot` (foliar phase before pycnidia eruption).
4. **`Wheat___Brown_rust`** (F1: 0.785)
   - *Primary Confusion Partner*: `Wheat___Yellow_rust` (scattered orange-brown pustules vs linear yellow stripes in juvenile infections).
5. **`Cotton___Bacterial_blight`** (F1: 0.794)
   - *Primary Confusion Partner*: `Cotton___Fusarium_wilt` (vascular leaf margin scorching).

---

## Confidence Calibration & Uncertainty Handling

### 1. Temperature Scaling
- Uncalibrated neural networks are prone to overconfident misclassifications.
- Post-hoc temperature scaling optimizes a single parameter $T = 1.25$ via Negative Log Likelihood (NLL) on the validation set.
- **Expected Calibration Error (ECE)** dropped from **7.8%** down to **1.8%**.

### 2. Decision Support Certainty Tiers
Configurable in `config.yaml`:
- **High Certainty ($\ge 0.85$)**: Reliable prediction; definitive prevention and agronomic measures presented.
- **Medium Certainty ($0.60 \le p < 0.85$)**: Plausible prediction; recommendation to check physical lesions.
- **Low Certainty ($< 0.60$)**: Returned with status `low_confidence`; prompts user to retake photo or consult local agronomist.

### 3. Out-Of-Distribution (OOD / Not-a-leaf) Filtering
- Uses Free Energy score $E(x) = -T \cdot \log \sum_{i} \exp(z_i / T)$ calibrated at the 95th percentile ($E_{\text{thresh}} = -4.80$).
- Non-leaf images (hands, soils, documents, irrelevant objects) trigger status `not_a_leaf`.

### 4. Image Quality Screening
- **Blur Detection**: OpenCV Laplacian variance $< 100.0$ returns `poor_image_quality` ("Image is blurry. Please hold camera steady.").
- **Illumination Screening**: Mean grayscale brightness $< 40.0$ (underexposed) or $> 225.0$ (glare/overexposed) returns `poor_image_quality`.

---

## Limitations and Ethical Considerations
- **Decision Support Only**: This system does NOT provide a legally binding or clinical diagnosis. All API responses include an agronomic disclaimer.
- **Single-Leaf Focus**: The model is trained on close-up single-leaf photos; canopy-level drone aerial shots will fail quality or OOD checks.
- **Chemical Treatment Risk**: Farmers are advised never to apply toxic or high-cost chemicals without corroborating field symptoms.
