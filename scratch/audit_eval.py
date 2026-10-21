import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, "ml_service")
from predict import CropDiseasePredictor

print("="*60)
print("AUDIT CHECK 4 & 5: ACCURACY, CONFUSION MATRIX, REAL PHOTOS")
print("="*60)

for backend_mode in ["huggingface", "legacy"]:
    os.environ["MODEL_BACKEND"] = backend_mode
    print(f"\n{'='*20} TESTING BACKEND: {backend_mode.upper()} {'='*20}")
    try:
        predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")
        print(f"[+] Successfully initialized backend '{backend_mode}'")
        print(f"    Classes: {predictor.num_classes}, Standard crops: {len(predictor.standard_crops)}")
        print(f"    Temperature: {predictor.temperature}, Confidence Threshold: {predictor.confidence_threshold}")
    except Exception as e:
        print(f"[!] FAILED to initialize backend '{backend_mode}': {e}")
        continue

    # 1. Test on test_lab (held-out lab images)
    lab_manifest = "data/manifests/test_lab_manifest.csv"
    if os.path.exists(lab_manifest):
        lab_df = pd.read_csv(lab_manifest)
        lab_correct_crop = 0
        lab_correct_disease = 0
        lab_valid = 0
        for idx, row in lab_df.iterrows():
            img_p = row["path"]
            true_label = row["label"]
            true_crop = true_label.split("___")[0].replace("_", " ")
            if not os.path.exists(img_p):
                continue
            with open(img_p, "rb") as f:
                img_bytes = f.read()
            res = predictor.predict(img_bytes, crop_hint=None, use_priors=False)
            lab_valid += 1
            pred_crop = res.get("crop", "").replace("_", " ")
            pred_disease = res.get("disease", "").replace("_", " ")
            pred_full = f"{res.get('crop')}___{res.get('disease')}".replace(" ", "_")
            if pred_crop.lower() in true_crop.lower() or true_crop.lower() in pred_crop.lower():
                lab_correct_crop += 1
            if pred_full.lower() == true_label.lower():
                lab_correct_disease += 1
        print(f"\n[+] Held-out LAB-style Test Set ({lab_valid} images):")
        print(f"    Crop Accuracy: {lab_correct_crop}/{lab_valid} ({lab_correct_crop/max(lab_valid,1):.2%})")
        print(f"    Full Class Accuracy: {lab_correct_disease}/{lab_valid} ({lab_correct_disease/max(lab_valid,1):.2%})")

    # 2. Test on test_field (held-out field images)
    field_manifest = "data/manifests/test_field_manifest.csv"
    if os.path.exists(field_manifest):
        field_df = pd.read_csv(field_manifest)
        field_correct_crop = 0
        field_correct_disease = 0
        field_valid = 0
        y_true_field = []
        y_pred_field = []
        for idx, row in field_df.iterrows():
            img_p = row["path"]
            true_label = row["label"]
            true_crop = true_label.split("___")[0].replace("_", " ")
            if not os.path.exists(img_p):
                continue
            with open(img_p, "rb") as f:
                img_bytes = f.read()
            res = predictor.predict(img_bytes, crop_hint=None, use_priors=False)
            field_valid += 1
            pred_crop = res.get("crop", "").replace("_", " ")
            pred_disease = res.get("disease", "").replace("_", " ")
            pred_full = f"{res.get('crop')}___{res.get('disease')}".replace(" ", "_")
            y_true_field.append(true_crop.lower())
            y_pred_field.append(pred_crop.lower())
            if pred_crop.lower() in true_crop.lower() or true_crop.lower() in pred_crop.lower():
                field_correct_crop += 1
            if pred_full.lower() == true_label.lower():
                field_correct_disease += 1
        print(f"\n[+] Held-out FIELD-style Test Set ({field_valid} images):")
        print(f"    Crop Accuracy: {field_correct_crop}/{field_valid} ({field_correct_crop/max(field_valid,1):.2%})")
        print(f"    Full Class Accuracy: {field_correct_disease}/{field_valid} ({field_correct_disease/max(field_valid,1):.2%})")

    # 3. Test on 30+ real phone photos in data/sample_real_leaves
    sample_dir = "data/sample_real_leaves"
    from eval_real import KNOWN_SAMPLE_ALIASES, parse_label_from_filename
    real_images = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    real_crop_correct = 0
    real_disease_correct = 0
    real_valid = 0
    confusions = Counter()
    y_true_all = []
    y_pred_all = []
    
    for img_p in real_images:
        true_label = parse_label_from_filename(img_p, predictor.classes)
        if not true_label:
            continue
        true_crop = true_label.split("___")[0].replace("_", " ")
        true_disease = true_label.split("___")[1].replace("_", " ") if "___" in true_label else ""
        
        with open(img_p, "rb") as f:
            img_bytes = f.read()
            
        res = predictor.predict(img_bytes, crop_hint=None, use_priors=False)
        real_valid += 1
        pred_crop = res.get("crop", "").replace("_", " ")
        pred_disease = res.get("disease", "").replace("_", " ")
        pred_full = f"{res.get('crop')}___{res.get('disease')}".replace(" ", "_")
        status = res.get("status")
        
        is_crop_match = (pred_crop.lower() in true_crop.lower() or true_crop.lower() in pred_crop.lower())
        is_disease_match = (pred_full.lower() == true_label.lower())
        
        if is_crop_match:
            real_crop_correct += 1
        if is_disease_match:
            real_disease_correct += 1
        else:
            confusions[(true_label, pred_full)] += 1
            
        y_true_all.append(true_crop.lower())
        y_pred_all.append(pred_crop.lower())

    print(f"\n[+] Real Phone Photos in data/sample_real_leaves ({real_valid} images):")
    print(f"    Crop Accuracy: {real_crop_correct}/{real_valid} ({real_crop_correct/max(real_valid,1):.2%})")
    print(f"    Full Class Accuracy: {real_disease_correct}/{real_valid} ({real_disease_correct/max(real_valid,1):.2%})")
    print(f"\n    Top 8 Class Confusions on Real Photos:")
    for (t, p), cnt in confusions.most_common(8):
        print(f"      True: {t:30s} -> Predicted: {p:30s} (count: {cnt})")
