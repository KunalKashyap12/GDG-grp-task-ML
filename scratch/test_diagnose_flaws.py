import os
import sys
import glob

# Ensure ml_service is in sys.path
sys.path.insert(0, os.path.abspath("ml_service"))
from predict import CropDiseasePredictor

predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

print(f"Loaded backend: {predictor.backend}")
print(f"Num classes: {predictor.num_classes}")
print(f"Standard crops: {predictor.standard_crops}")

# Test real sample leaves
samples = [
    "data/sample_real_leaves/Bell_Pepper___Bacterial_spot_0.jpg",
    "data/sample_real_leaves/real_pepper_bacterial_spot.jpg",
    "data/sample_real_leaves/a-bellpepper-leaf.png",
    "data/sample_real_leaves/Tomato___Septoria_leaf_spot_0.jpg",
    "data/sample_real_leaves/Potato___Early_blight_0.jpg",
    "data/sample_real_leaves/Apple___Apple_scab_0.jpg",
    "data/sample_real_leaves/Grape___Black_rot_0.jpg",
    "data/sample_real_leaves/Maize___Common_rust_0.jpg"
]

results = []
for s in samples:
    if not os.path.exists(s):
        print(f"Missing sample: {s}")
        continue
    with open(s, "rb") as f:
        img_bytes = f.read()
    
    # 1. Unconstrained
    res = predictor.predict(img_bytes)
    # Extract morphology
    morph = res.get("morphology", {})
    results.append({
        "file": os.path.basename(s),
        "status": res["status"],
        "pred_crop": res["crop"],
        "pred_disease": res["disease"],
        "confidence": res["confidence"],
        "morph_shape": morph.get("shape"),
        "morph_margin": morph.get("margin"),
        "morph_geometry": morph.get("geometry_class"),
        "top3": [(t["label"], t["score"]) for t in res.get("top3", [])]
    })

import json
print(json.dumps(results, indent=2))
