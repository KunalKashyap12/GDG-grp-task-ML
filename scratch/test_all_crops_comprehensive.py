import os
import sys
import glob
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath("ml_service"))
from predict import CropDiseasePredictor

predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

sample_dir = "data/sample_real_leaves"
image_files = sorted(glob.glob(os.path.join(sample_dir, "*.*")))
image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Total images found in {sample_dir}: {len(image_files)}")

crop_results = defaultdict(list)

for img_path in image_files:
    fname = os.path.basename(img_path)
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    
    # Extract expected crop from filename
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
    morph = res.get("morphology", {}) or {}

    crop_results[expected_crop or "Other"].append({
        "file": fname,
        "pred_crop": res.get("crop"),
        "pred_disease": res.get("disease"),
        "status": res.get("status"),
        "confidence": res.get("confidence"),
        "morph_shape": morph.get("shape"),
        "morph_margin": morph.get("margin"),
        "morph_solidity": morph.get("solidity"),
        "morph_roughness": morph.get("roughness"),
        "morph_is_linear": morph.get("is_linear_blade"),
        "top3": [(t["label"], t["score"]) for t in res.get("top3", [])]
    })

summary = {}
for crop, items in crop_results.items():
    total = len(items)
    # Check crop match
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
                "conf": it["confidence"],
                "status": it["status"],
                "morph": f"{it['morph_shape']} | {it['morph_margin']}"
            }
            for it in items[:3]
        ]
    }

print("\n=== SUMMARY ACROSS ALL CROP CATEGORIES ===")
print(json.dumps(summary, indent=2))
