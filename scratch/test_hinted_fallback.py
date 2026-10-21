import os
import sys

sys.path.insert(0, os.path.abspath("ml_service"))
from predict import CropDiseasePredictor

predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

test_cases = [
    ("data/sample_real_leaves/Wheat___Yellow_rust_0.jpg", "Wheat"),
    ("data/sample_real_leaves/a-cotton-leaf.png", "Cotton"),
    ("data/sample_real_leaves/a-rice-leaf.png", "Rice"),
    ("data/sample_real_leaves/a-sugarcane-leaf.png", "Sugarcane"),
    ("data/sample_real_leaves/real_pepper_bacterial_spot.jpg", None),
    ("data/sample_real_leaves/real_pepper_bacterial_spot.jpg", "Bell_Pepper")
]

for path, hint in test_cases:
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        img_bytes = f.read()
    res = predictor.predict(img_bytes, crop_hint=hint)
    print(f"\nFile: {os.path.basename(path)} | Hint: {hint}")
    print(f"Status: {res['status']} | Crop: {res['crop']} | Disease: {res['disease']} | Confidence: {res['confidence']}")
    print(f"Top 3: {[(t['label'], t['score']) for t in res.get('top3', [])]}")
