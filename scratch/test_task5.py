import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tempfile
import shutil
import pandas as pd
import numpy as np
from PIL import Image

import importlib
filter_and_split = importlib.import_module("ml_training.02_filter_and_split")
compute_all_perceptual_hashes = filter_and_split.compute_all_perceptual_hashes

evaluate = importlib.import_module("ml_training.04_evaluate")
generate_source_class_report = evaluate.generate_source_class_report

def test_rotation_deduplication():
    print("Testing compute_all_perceptual_hashes and rotation deduplication...")
    temp_dir = tempfile.mkdtemp()
    try:
        # Create an asymmetric test image
        arr = np.zeros((120, 120, 3), dtype=np.uint8)
        arr[20:50, 20:60] = [210, 80, 40]
        arr[70:110, 60:100] = [30, 180, 50]
        orig_img = Image.fromarray(arr)
        orig_path = os.path.join(temp_dir, "orig.jpg")
        orig_img.save(orig_path)

        # Create 90-degree rotated copy
        rot90_img = orig_img.transpose(Image.ROTATE_90)
        rot90_path = os.path.join(temp_dir, "rot90.jpg")
        rot90_img.save(rot90_path)

        # Create horizontal flip copy
        flip_img = orig_img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_path = os.path.join(temp_dir, "flip.jpg")
        flip_img.save(flip_path)

        # Compute hashes
        h_orig, orig_hashes = compute_all_perceptual_hashes(orig_path)
        h_rot90, rot_hashes = compute_all_perceptual_hashes(rot90_path)
        h_flip, flip_hashes = compute_all_perceptual_hashes(flip_path)

        # Hamming distance between original hash and rotated hash is large (>20)
        dist_direct = h_orig - h_rot90
        print(f"Direct pHash distance between original and 90-deg rotated: {dist_direct}")
        assert dist_direct > 4, "Direct pHash should miss rotated image without rotation check!"

        # But with all_hashes check:
        match_found = False
        for cand in rot_hashes:
            for seen in orig_hashes:
                if (cand - seen) <= 4:
                    match_found = True
                    break
            if match_found:
                break
        print("Rotation hash match found:", match_found)
        assert match_found, "Rotated image MUST match when using all_hashes!"

        # And horizontal flip match:
        flip_match = False
        for cand in flip_hashes:
            for seen in orig_hashes:
                if (cand - seen) <= 4:
                    flip_match = True
                    break
            if flip_match:
                break
        print("Flip hash match found:", flip_match)
        assert flip_match, "Flipped image MUST match when using all_hashes!"

        print("[PASS] Rotation & flip deduplication verified!")
    finally:
        shutil.rmtree(temp_dir)

def test_shortcut_risk_report():
    print("\nTesting per-source and per-class shortcut learning CSV report...")
    temp_dir = tempfile.mkdtemp()
    try:
        # Create synthetic manifest
        data = [
            # Class A: 95% PlantVillage, 5% PlantDoc -> Should be flagged as SHORTCUT RISK!
            *([{"label": "Tomato___Bacterial_spot", "source": "PlantVillage"}] * 95),
            *([{"label": "Tomato___Bacterial_spot", "source": "PlantDoc"}] * 5),
            # Class B: 50% PlantVillage, 50% PlantDoc -> Balanced!
            *([{"label": "Tomato___Early_blight", "source": "PlantVillage"}] * 50),
            *([{"label": "Tomato___Early_blight", "source": "PlantDoc"}] * 50),
            # Class C: 100% Kaggle_Rice -> Flagged!
            *([{"label": "Rice___Brown_spot", "source": "Kaggle_Rice"}] * 40),
        ]
        manifest_df = pd.DataFrame(data)
        class_names = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Rice___Brown_spot"]

        csv_path = os.path.join(temp_dir, "test_report.csv")
        report_df, flagged = generate_source_class_report(manifest_df, class_names, output_csv_path=csv_path)

        assert os.path.exists(csv_path), "CSV report should have been written!"
        print(f"Report rows: {len(report_df)}")
        print(f"Flagged classes count: {len(flagged)}")

        flagged_classes = [f["class"] for f in flagged]
        print("Flagged classes:", flagged_classes)
        assert "Tomato___Bacterial_spot" in flagged_classes, "Tomato___Bacterial_spot (>90%) MUST be flagged!"
        assert "Rice___Brown_spot" in flagged_classes, "Rice___Brown_spot (100%) MUST be flagged!"
        assert "Tomato___Early_blight" not in flagged_classes, "Tomato___Early_blight (50/50) should NOT be flagged!"

        # Check CSV content
        saved_df = pd.read_csv(csv_path)
        assert "dominant_source" in saved_df.columns
        assert "shortcut_risk" in saved_df.columns
        print("[PASS] Shortcut risk CSV report verified!")
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_rotation_deduplication()
    test_shortcut_risk_report()
    print("\nALL TASK 5 TESTS PASSED SUCCESSFULLY!")
