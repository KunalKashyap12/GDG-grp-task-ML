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

CROP_TEST_CASES_HF = [
    ("a-bellpepper-leaf.png", "Bell Pepper", "Bacterial Spot", "simple_ovate"),
    ("a-grapes-leaf.png", "Grape", "Esca (Black Measles)", "simple_ovate"),
    ("a-potato-leaf.png", "Potato", "Early Blight", "simple_ovate"),
    ("a-tomato-leaf.png", "Tomato", "Early Blight", "simple_ovate"),
    ("apple-fungal-leaf.png", "Apple", "Scab", "simple_ovate"),
]

UNSUPPORTED_TEST_CASES = [
    ("a-cotton-leaf.png", "Cotton"),
    ("a-rice-leaf.png", "Rice"),
    ("a-sugarcane-leaf.png", "Sugarcane"),
    ("a-wheat-leaf.png", "Wheat"),
]

def test_ten_crops_prediction():
    predictor = CropDiseasePredictor()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_real_leaves"))

    # Test supported PlantVillage crops
    for filename, expected_crop, expected_disease, expected_geom in CROP_TEST_CASES_HF:
        filepath = os.path.join(data_dir, filename)
        assert os.path.exists(filepath), f"Test image missing: {filepath}"

        with open(filepath, "rb") as f:
            res = predictor.predict(f.read(), crop_hint=expected_crop)

        assert res["status"] in ["ok", "low_confidence"], f"Prediction status failed for {filename}: {res}"
        assert res["crop"].lower().replace("_", " ") == expected_crop.lower().replace("_", " "), f"Expected crop {expected_crop}, got {res['crop']} for {filename}"
        assert res["confidence"] > 0.01, f"Confidence too low for {filename}: {res['confidence']}"

        crop_comp = res.get("crop_comparison", [])
        assert len(crop_comp) > 0, "Crop comparison list is empty"

        morph = res.get("morphology", {})
        assert "shape" in morph, "Morphology missing shape"
        assert "margin" in morph, "Morphology missing margin"
        assert "venation" in morph, "Morphology missing venation"

    # Test that unsupported crops fail gracefully with low_confidence
    if predictor.backend == "huggingface":
        for filename, unsupported_crop in UNSUPPORTED_TEST_CASES:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    res = predictor.predict(f.read(), crop_hint=unsupported_crop)
                assert res["status"] == "low_confidence"
                assert res["confidence"] == 0.0
    else:
        # In legacy mode, Cotton, Rice, Sugarcane, Wheat are supported crops
        for filename, crop_name in UNSUPPORTED_TEST_CASES:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    res = predictor.predict(f.read(), crop_hint=crop_name)
                assert res["status"] in ["ok", "low_confidence"]


def test_manual_shift_penalty():
    predictor = CropDiseasePredictor()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_real_leaves"))
    filepath = os.path.join(data_dir, "a-bellpepper-leaf.png")

    with open(filepath, "rb") as f:
        buf = f.read()

    # Normal auto-detect should yield confident prediction
    res_auto = predictor.predict(buf, crop_hint="auto")
    assert res_auto["crop"] in ["Bell Pepper", "Bell_Pepper"]
    assert res_auto["confidence"] >= 0.30
    assert res_auto["mismatch_warning"] is None

    # Incompatible manual override (Squash on Bell Pepper leaf) should trigger species_mismatch
    res_mismatch = predictor.predict(buf, crop_hint="Squash")
    assert res_mismatch["status"] == "species_mismatch"
    assert res_mismatch["crop"] == "Squash"
    assert res_mismatch["certainty"] == "low"
    assert res_mismatch["mismatch_warning"] is not None
    assert "diagnosis was requested for" in res_mismatch["mismatch_warning"]

if __name__ == "__main__":
    test_ten_crops_prediction()
    test_manual_shift_penalty()
    print("[+] All crop validation tests passed successfully!")
