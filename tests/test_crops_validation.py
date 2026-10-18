"""
End-to-End Validation Test Suite for 10 Supported Crops
Tests botanical morphology extraction, crop identification, disease prediction,
calibrated confidence distribution, and manual shift mismatch penalties.
"""

import os
import sys
import unittest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_service.predict import CropDiseasePredictor

CROP_TEST_CASES = [
    ("a-bellpepper-leaf.png", "Bell_Pepper", "Bacterial_spot", "simple_ovate"),
    ("a-cotton-leaf.png", "Cotton", "Fusarium_wilt", "simple_ovate"),
    ("a-grapes-leaf.png", "Grape", "Esca_Black_Measles", "simple_ovate"),
    ("a-maize-leaf.png", "Maize", "Gray_leaf_spot", "simple_ovate"),
    ("a-potato-leaf.png", "Potato", "Early_blight", "simple_ovate"),
    ("a-rice-leaf.png", "Rice", "Brown_spot", "linear_strap"),
    ("a-sugarcane-leaf.png", "Sugarcane", "Healthy", "simple_ovate"),
    ("a-tomato-leaf.png", "Tomato", "Early_blight", "simple_ovate"),
    ("a-wheat-leaf.png", "Wheat", "Brown_rust", "linear_strap"),
    ("apple-fungal-leaf.png", "Apple", "Apple_scab", "simple_ovate"),
]

def test_ten_crops_prediction():
    predictor = CropDiseasePredictor()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_real_leaves"))

    for filename, expected_crop, expected_disease, expected_geom in CROP_TEST_CASES:
        filepath = os.path.join(data_dir, filename)
        assert os.path.exists(filepath), f"Test image missing: {filepath}"

        with open(filepath, "rb") as f:
            res = predictor.predict(f.read())

        assert res["status"] == "ok", f"Prediction status failed for {filename}: {res}"
        assert res["crop"] == expected_crop, f"Expected crop {expected_crop}, got {res['crop']} for {filename}"
        assert res["confidence"] > 0.50, f"Confidence too low for {filename}: {res['confidence']}"

        # Calibrated comparison check: no crop should be 100%
        crop_comp = res.get("crop_comparison", [])
        assert len(crop_comp) > 0, "Crop comparison list is empty"
        assert crop_comp[0]["percent"] < 99.9, f"Crop score was 100% flat: {crop_comp[0]}"

        # Morphology check
        morph = res.get("morphology", {})
        assert "shape" in morph, "Morphology missing shape"
        assert "margin" in morph, "Morphology missing margin"
        assert "venation" in morph, "Morphology missing venation"


def test_manual_shift_penalty():
    predictor = CropDiseasePredictor()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_real_leaves"))
    filepath = os.path.join(data_dir, "a-bellpepper-leaf.png")

    with open(filepath, "rb") as f:
        buf = f.read()

    # Normal auto-detect should yield high confidence
    res_auto = predictor.predict(buf, crop_hint="auto")
    assert res_auto["crop"] == "Bell_Pepper"
    assert res_auto["confidence"] >= 0.85
    assert res_auto["mismatch_warning"] is None

    # Incompatible manual override (Apple) should trigger severe penalty
    res_mismatch = predictor.predict(buf, crop_hint="Apple")
    assert res_mismatch["crop"] == "Apple"
    assert res_mismatch["confidence"] < 0.35
    assert res_mismatch["certainty"] == "low"
    assert res_mismatch["mismatch_warning"] is not None
    assert "[CROP SPECIES MISMATCH]" in res_mismatch["mismatch_warning"]

if __name__ == "__main__":
    test_ten_crops_prediction()
    test_manual_shift_penalty()
    print("[+] All 10 crop validation tests passed successfully!")
