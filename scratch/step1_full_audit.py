import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, "ml_service")
from predict import CropDiseasePredictor
from eval_real import KNOWN_SAMPLE_ALIASES, parse_label_from_filename

print("="*70)
print("COMPREHENSIVE AUDIT REPORT FOR STEP 1")
print("="*70)

# Evaluate Active Hugging Face Model (14 crops, 38 classes)
os.environ["MODEL_BACKEND"] = "huggingface"
predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

sample_dir = "data/sample_real_leaves"
real_images = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

y_true_crop = []
y_pred_crop = []
y_true_class = []
y_pred_class = []

# Only evaluate on supported crops for the active model (14 crops)
active_supported = [c.lower() for c in predictor.standard_crops]
filtered_images = []

for img_p in real_images:
    true_label = parse_label_from_filename(img_p, predictor.classes)
    if not true_label:
        continue
    raw_crop = true_label.split("___")[0].replace("_", " ")
    
    # Check if this crop is one of the 14 crops
    matched_std = None
    for std_c in predictor.standard_crops:
        if raw_crop.lower() in std_c.lower() or std_c.lower() in raw_crop.lower():
            matched_std = std_c
            break
    if not matched_std:
        continue # Skip legacy crops (Wheat, Rice, Cotton, Sugarcane) for active model audit
        
    filtered_images.append((img_p, matched_std, true_label))

print(f"\n[1] Evaluating ACTIVE Model (HuggingFace MobileNetV2, 14 crops) on {len(filtered_images)} real photos...")
for img_p, true_crop, true_label in filtered_images:
    with open(img_p, "rb") as f:
        img_bytes = f.read()
    res = predictor.predict(img_bytes, crop_hint=None, use_priors=False)
    pred_crop = res.get("crop", "Unknown")
    pred_disease = res.get("disease", "Unknown")
    pred_class = f"{pred_crop}___{pred_disease}".replace(" ", "_")
    
    y_true_crop.append(true_crop)
    y_pred_crop.append(pred_crop)
    y_true_class.append(true_label)
    y_pred_class.append(pred_class)

crop_correct = sum(1 for yt, yp in zip(y_true_crop, y_pred_crop) if yt.lower() == yp.lower() or yt.lower() in yp.lower())
class_correct = sum(1 for yt, yp in zip(y_true_class, y_pred_class) if yt.lower() == yp.lower())

print(f"\nACTIVE MODEL PERFORMANCE ON REAL PHOTOS (N={len(filtered_images)}):")
print(f"  Overall Crop Accuracy:       {crop_correct}/{len(filtered_images)} ({crop_correct/len(filtered_images):.2%})")
print(f"  Overall Full Class Accuracy: {class_correct}/{len(filtered_images)} ({class_correct/len(filtered_images):.2%})")

# Per-crop breakdown
print("\n--- PER-CROP ACCURACY BREAKDOWN ---")
crop_stats = {}
for yt, yp, yc_t, yc_p in zip(y_true_crop, y_pred_crop, y_true_class, y_pred_class):
    if yt not in crop_stats:
        crop_stats[yt] = {"total": 0, "crop_correct": 0, "class_correct": 0, "confused_with": []}
    crop_stats[yt]["total"] += 1
    if yt.lower() == yp.lower() or yt.lower() in yp.lower():
        crop_stats[yt]["crop_correct"] += 1
    else:
        crop_stats[yt]["confused_with"].append(yp)
    if yc_t.lower() == yc_p.lower():
        crop_stats[yt]["class_correct"] += 1

for c, s in sorted(crop_stats.items()):
    c_acc = s["crop_correct"] / s["total"]
    f_acc = s["class_correct"] / s["total"]
    conf = f" -> Confused as: {dict(Counter(s['confused_with']))}" if s["confused_with"] else ""
    print(f"  {c:15s}: N={s['total']:2d} | Crop Acc: {c_acc:6.1%} | Full Disease Acc: {f_acc:6.1%}{conf}")

# Top Cross-Crop and Within-Crop Confusions
print("\n--- TOP CONFUSIONS ON REAL USER PHOTOS ---")
conf_counter = Counter()
for yt, yp in zip(y_true_class, y_pred_class):
    if yt.lower() != yp.lower():
        conf_counter[(yt, yp)] += 1
for (t, p), count in conf_counter.most_common(10):
    t_crop = t.split("___")[0]
    p_crop = p.split("___")[0]
    tag = "[CROSS-CROP]" if t_crop.lower() != p_crop.lower() else "[SAME-CROP]"
    print(f"  {tag:12s} True: {t:32s} -> Predicted: {p:32s} ({count}x)")
