"""
Dataset Evaluation Script
Evaluates a folder of class-named images, prints per-class precision and recall,
the confusion matrix, and the 10 most confused class pairs.
Reuses the ML service's inference engine and preprocessing function.
"""

import os
import re
import sys
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List

# Ensure ml_service and repo root are in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
ML_SERVICE_DIR = os.path.join(REPO_ROOT, "ml_service")
for p in (BASE_DIR, REPO_ROOT, ML_SERVICE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import CropDiseasePredictor, preprocess_image, clean_token


def get_ground_truth(filepath: str) -> str:
    """Extracts Crop___Disease ground truth from filename or parent directory."""
    fname = os.path.basename(filepath)
    parent = os.path.basename(os.path.dirname(filepath))

    # Check parent folder format: e.g. Tomato___Early_blight/xxx.jpg
    if "___" in parent:
        return parent

    # Check filename format: e.g. Tomato___Early_blight_0.jpg or a-grapes-leaf.png
    stem = os.path.splitext(fname)[0]
    if "___" in stem:
        match = re.match(r"^([A-Za-z0-9_]+___[A-Za-z0-9_]+?)(?:_\d+)?$", stem)
        if match:
            return match.group(1)
        return stem

    # Mapping for sample leaf images
    sample_map = {
        "a-grapes-leaf": "Grape___Esca_Black_Measles",
        "a-potato-leaf": "Potato___Early_blight",
        "a-tomato-leaf": "Tomato___Early_blight",
        "a-bellpepper-leaf": "Bell_Pepper___Bacterial_spot",
        "apple-fungal-leaf": "Apple___Apple_scab",
        "a-cotton-leaf": "Cotton___Bacterial_blight",
        "a-rice-leaf": "Rice___Brown_spot",
        "a-sugarcane-leaf": "Sugarcane___Red_rot",
        "a-wheat-leaf": "Wheat___Yellow_rust",
        "a-maize-leaf": "Maize___Common_rust",
        "real_apple_cedar_rust": "Apple___Cedar_apple_rust",
        "real_apple_scab": "Apple___Apple_scab",
        "real_grape_black_rot": "Grape___Black_rot",
        "real_maize_common_rust": "Maize___Common_rust",
        "real_pepper_bacterial_spot": "Bell_Pepper___Bacterial_spot",
        "real_potato_early_blight": "Potato___Early_blight",
        "real_tomato_early_blight": "Tomato___Early_blight",
        "real_tomato_healthy": "Tomato___Healthy",
        "real_tomato_late_blight": "Tomato___Late_blight",
        "Tomato_Early_Blight_REAL": "Tomato___Early_blight",
        "user_apple_leaf": "Apple___Apple_scab",
    }
    return sample_map.get(stem, stem)


def evaluate(data_dir: str, config_path: str = "config.yaml"):
    predictor = CropDiseasePredictor(config_path=config_path)
    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}

    image_files: List[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported_exts:
                image_files.append(os.path.join(root, f))

    if not image_files:
        print(f"[!] No valid image files found in '{data_dir}'.")
        return

    print(f"[*] Evaluating {len(image_files)} images from '{data_dir}'...")

    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_classes = set()

    for path in sorted(image_files):
        true_cls = get_ground_truth(path)
        with open(path, "rb") as f:
            buf = f.read()
        res = predictor.predict(buf)
        crop_clean = res.get("crop", "unknown").replace(" ", "_")
        disease_clean = res.get("disease", "unknown").replace(" ", "_")
        pred_cls = f"{crop_clean}___{disease_clean}"

        matrix[true_cls][pred_cls] += 1
        all_classes.add(true_cls)
        all_classes.add(pred_cls)

    # 1. Per-Class Precision and Recall
    print("\n" + "=" * 70)
    print(f"{'Class Name':<42} | {'Precision':<10} | {'Recall':<10} | {'Support':<8}")
    print("-" * 70)

    total_tp = 0
    total_samples = 0

    for cls in sorted(all_classes):
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in all_classes if other != cls)
        fn = sum(matrix[cls][other] for other in all_classes if other != cls)
        support = tp + fn

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / support) if support > 0 else 0.0

        if support > 0:
            total_tp += tp
            total_samples += support
            print(f"{cls:<42} | {precision:>9.2%} | {recall:>9.2%} | {support:>7}")

    accuracy = (total_tp / total_samples) if total_samples > 0 else 0.0
    print("-" * 70)
    print(f"Overall Accuracy: {accuracy:.2%} ({total_tp}/{total_samples} correct)\n")

    # 2. Top 10 Most Confused Class Pairs
    confused_pairs = []
    for true_cls in all_classes:
        for pred_cls, count in matrix[true_cls].items():
            if true_cls != pred_cls and count > 0:
                confused_pairs.append((true_cls, pred_cls, count))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print("=" * 70)
    print("TOP 10 MOST CONFUSED CLASS PAIRS (True -> Predicted):")
    print("-" * 70)
    if not confused_pairs:
        print("  None (Perfect classification!)")
    else:
        for idx, (t_cls, p_cls, cnt) in enumerate(confused_pairs[:10], start=1):
            print(f"  {idx:>2}. {t_cls:<32} -> {p_cls:<32} : {cnt:>3} errors")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dataset folder using ML service inference.")
    parser.add_argument("--data_dir", type=str, default="data/sample_real_leaves", help="Path to image folder")
    parser.add_argument("--config", type=str, default="ml_service/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    evaluate(args.data_dir, args.config)
