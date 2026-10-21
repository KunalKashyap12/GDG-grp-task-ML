import os
import sys
import glob
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath("ml_service"))

# Temporarily set backend to legacy
os.environ["MODEL_BACKEND"] = "legacy"

from predict import CropDiseasePredictor

predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

print(f"Loaded backend: {predictor.backend}")
print(f"Num classes: {predictor.num_classes}")
print(f"Standard crops: {predictor.standard_crops}")

sample_dir = "data/sample_real_leaves"
image_files = sorted(glob.glob(os.path.join(sample_dir, "*.*")))
image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

crop_results = defaultdict(list)

for img_path in image_files:
    fname = os.path.basename(img_path)
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    
    expected_crop = None
    for c in ["Apple", "Bell_Pepper", "Cotton", "Grape", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"]:
        if fname.lower().startswith(c.lower()):
            expected_crop = c
            break
    if not expected_crop:
        if "bellpepper" in fname.lower() or "pepper" in fname.lower():
            expected_crop = "Bell_Pepper"
        elif "cotton" in fname.lower():
            expected_crop = "Cotton"
        elif "grape" in fname.lower():
            expected_crop = "Grape"
        elif "maize" in fname.lower():
            expected_crop = "Maize"
        elif "potato" in fname.lower():
            expected_crop = "Potato"
        elif "rice" in fname.lower():
            expected_crop = "Rice"
        elif "sugarcane" in fname.lower():
            expected_crop = "Sugarcane"
        elif "tomato" in fname.lower():
            expected_crop = "Tomato"
        elif "wheat" in fname.lower():
            expected_crop = "Wheat"
        elif "apple" in fname.lower():
            expected_crop = "Apple"

    res = predictor.predict(img_bytes)
    crop_results[expected_crop or "Other"].append({
        "file": fname,
        "pred_crop": res.get("crop"),
        "pred_disease": res.get("disease"),
        "status": res.get("status"),
        "confidence": res.get("confidence")
    })

summary = {}
for crop, items in crop_results.items():
    total = len(items)
    correct_crop = 0
    for it in items:
        p = (it["pred_crop"] or "").lower().replace("_", "").replace(" ", "").replace(",", "")
        exp = crop.lower().replace("_", "").replace(" ", "").replace(",", "")
        if exp == "maize" and p == "corn(maize)":
            correct_crop += 1
        elif exp == p or exp in p or p in exp:
            correct_crop += 1
    summary[crop] = {
        "total": total,
        "correct_crop": correct_crop,
        "accuracy": round(correct_crop / total * 100, 1) if total > 0 else 0,
        "samples": [
            {
                "file": it["file"],
                "pred": f"{it['pred_crop']} - {it['pred_disease']}",
                "conf": it["confidence"]
            }
            for it in items[:3]
        ]
    }

print("\n=== SUMMARY ACROSS ALL CROP CATEGORIES (LEGACY BACKEND) ===")
print(json.dumps(summary, indent=2))
