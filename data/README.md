# Crop Disease Dataset Architecture & Guide

This directory contains the dataset structure, manifests, and sample leaf photographs for the **Crop Disease Detection & Decision Support System**.

---

## 1. Directory Structure

The `data/` directory is organized modularly to prevent clutter and maintain strict separation between raw downloads, processed training partitions, metadata manifests, and sample evaluation images:

```
data/
├── manifests/              # Partition catalogs and mapping records (CSVs)
│   ├── manifest.csv        # Master unified dataset manifest
│   ├── train_manifest.csv  # Training partition split (70%)
│   ├── val_manifest.csv    # Validation partition split (15%)
│   ├── test_lab_manifest.csv   # Lab test split (PlantVillage benchmark)
│   └── test_field_manifest.csv # Field test split (PlantDoc benchmark)
│
├── processed/              # Standard PyTorch ImageFolder splits
│   ├── train/              # Subdirectories per Crop___Disease class
│   ├── val/                # Subdirectories per Crop___Disease class
│   ├── test_lab/           # Benchmark lab evaluation split
│   └── test_field/         # Benchmark real-world field evaluation split
│
├── raw/                    # Destination for downloaded external datasets
│   ├── PlantVillage/       # 6 Lab crops: Apple, Bell Pepper, Grape, Maize, Potato, Tomato
│   ├── PlantDoc/           # In-field real-world leaf dataset
│   ├── Kaggle_Rice/        # Rice leaf diseases (Blight, Brown spot, Blast)
│   ├── Kaggle_Wheat/       # Wheat rust & leaf diseases
│   ├── Kaggle_Cotton/      # Cotton bacterial blight, curl virus, fusarium wilt
│   └── Kaggle_Sugarcane/   # Sugarcane red rot, mosaic, healthy
│
└── sample_real_leaves/     # Real photographic leaf images for testing & inference
    ├── Tomato_Early_Blight_REAL.jpg
    ├── real_apple_cedar_rust.jpg
    ├── real_apple_scab.jpg
    ├── real_grape_black_rot.jpg
    ├── real_maize_common_rust.jpg
    ├── real_pepper_bacterial_spot.jpg
    ├── real_potato_early_blight.jpg
    ├── real_tomato_early_blight.jpg
    ├── real_tomato_healthy.jpg
    └── real_tomato_late_blight.jpg
```

---

## 2. Why Were There "Green Circles with Red Spots"?

If you previously encountered green circles with red/brown spots, here is why:
- In `ml_training/01_merge_datasets.py`, an initial mock generator (`create_synthetic_image()`) was included as a **fallback smoke-testing utility**.
- Its sole purpose was to allow developers to verify the Python pipeline, Docker setup, FastAPI microservice, and Spring Boot backend end-to-end immediately without waiting to download **4 to 8 Gigabytes** of raw dataset images.
- **These dummy drawings have now been completely removed** from all data folders.

---

## 3. What Images Are Used for Model Training?

> [!IMPORTANT]
> **Model training requires REAL leaf photographs.**
> Convolutional neural networks (like MobileNetV3) learn fine-grained visual patterns—such as necrotic lesions, chlorotic halos, fungal sporulation, and leaf venation—that do not exist in drawn circles.

For genuine model training, the system uses:
1. **PlantVillage Dataset**: Laboratory-condition photos on solid backgrounds (Apple, Bell Pepper, Grape, Maize, Potato, Tomato).
2. **PlantDoc Dataset**: In-field real-world photographs taken in actual agricultural environments under natural lighting, blur, and occlusions.
3. **Kaggle Agricultural Datasets**: Domain-specific field photography for Rice, Wheat, Cotton, and Sugarcane.

---

## 4. How to Download the Complete Real Datasets

To populate `data/raw/` with the complete real-world image datasets:

### Option A: Automated Download (Kaggle CLI)
1. Install Kaggle CLI and place your `kaggle.json` in `~/.kaggle/`:
   ```bash
   pip install kaggle
   ```
2. Run the downloader:
   ```bash
   python ml_training/download_datasets.py --download-all
   ```

### Option B: Direct Download Sources
- **PlantVillage**: https://github.com/spMohanty/PlantVillage-Dataset/archive/master.zip (Unzip to `data/raw/PlantVillage`)
- **PlantDoc**: https://github.com/pratikkayal/PlantDoc-Dataset/archive/master.zip (Unzip to `data/raw/PlantDoc`)
- **Kaggle Rice**: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
- **Kaggle Wheat**: https://www.kaggle.com/datasets/olyadivotkey/wheat-leaf-dataset
- **Kaggle Cotton**: https://www.kaggle.com/datasets/dhanushnarayananr/cotton-plant-disease
- **Kaggle Sugarcane**: https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset

---

## 5. Instant Testing with Real Leaf Samples

You can immediately test the FastAPI service (`localhost:8000/predict`) or Spring Boot API (`localhost:8080/api/predict`) using any genuine leaf photograph in `data/sample_real_leaves/`:

```bash
# Test FastAPI ML Service
curl -X POST -F "file=@data/sample_real_leaves/real_tomato_early_blight.jpg" http://localhost:8000/predict

# Test Spring Boot Backend
curl -X POST -F "image=@data/sample_real_leaves/real_tomato_early_blight.jpg" http://localhost:8080/api/predict
```
