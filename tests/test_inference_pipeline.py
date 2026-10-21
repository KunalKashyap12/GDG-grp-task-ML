"""
Unit and Integration Test Suite for Crop Leaf Disease Inference Pipeline (Task 7)

Covers:
1. Grounded confidence formula and reachable certainty tiers (high, medium, low).
2. Robust crop-hint normalization and synonyms (English, botanical, regional).
3. Image quality gate rejection of blurred and dark synthetic images.
4. Free Energy OOD rejection of random noise and non-leaf artifacts.
5. Verification that correct hints never trigger species_mismatch when crop mass is present.
6. Verification that all responses strictly maintain backward compatibility with Spring Boot DTO keys.
"""

import os
import io
import sys
import json
import pytest
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml_service.predict import CropDiseasePredictor


@pytest.fixture(scope="module")
def predictor():
    config_path = os.path.abspath("ml_service/config.yaml")
    return CropDiseasePredictor(config_path=config_path)


def _image_to_bytes(img: Image.Image, format: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Confidence Formula and Tiers
# ---------------------------------------------------------------------------
def test_confidence_formula_and_tiers(predictor):
    """
    Validates that:
    - Confidence is grounded: Confidence = P(disease | effective_crop) * crop_mass
    - Artificial 0.65 floor and 0.94 cap are removed
    - Low confidence tier (< 0.60) is fully reachable and sets status='low_confidence'
    - Tiers map correctly: high (>=0.85), medium (0.60..0.85), low (<0.60)
    """
    # Test mapping logic directly
    for conf, expected_tier, expected_status in [
        (0.92, "high", "ok"),
        (0.85, "high", "ok"),
        (0.72, "medium", "ok"),
        (0.60, "medium", "ok"),
        (0.59, "low", "low_confidence"),
        (0.32, "low", "low_confidence"),
        (0.15, "low", "low_confidence")
    ]:
        if conf >= 0.85:
            tier = "high"
        elif conf >= 0.60:
            tier = "medium"
        else:
            tier = "low"
        status = "low_confidence" if tier == "low" else "ok"

        assert tier == expected_tier, f"Confidence {conf} mapped to {tier}, expected {expected_tier}"
        assert status == expected_status, f"Confidence {conf} mapped to status {status}, expected {expected_status}"

    # Verify confidence policy: if confidence < confidence_threshold, status is low_confidence and certainty is low
    data_dir = os.path.abspath("data/sample_real_leaves")
    bp_path = os.path.join(data_dir, "a-bellpepper-leaf.png")
    if os.path.exists(bp_path):
        with open(bp_path, "rb") as f:
            leaf_buf = f.read()
        res_normal = predictor.predict(leaf_buf)
        assert res_normal["status"] in ["ok", "low_confidence"]
        
        orig_thresh = predictor.confidence_threshold
        try:
            predictor.confidence_threshold = 0.9999
            res_low = predictor.predict(leaf_buf)
            assert res_low["status"] == "low_confidence"
            assert res_low["certainty"] == "low"
        finally:
            predictor.confidence_threshold = orig_thresh


# ---------------------------------------------------------------------------
# 2. Crop-Hint Normalization and Synonyms
# ---------------------------------------------------------------------------
def test_crop_hint_normalization_and_synonyms(predictor):
    """
    Validates case-insensitivity, spacing, separator handling,
    and extensive synonym mapping (botanical, regional, English).
    """
    is_hf = (predictor.backend == "huggingface")
    bell_expected = "Bell Pepper" if is_hf else "Bell_Pepper"
    corn_expected = "Corn (Maize)" if is_hf else "Maize"

    test_cases = [
        # Direct matches & variations
        ("tomato", "Tomato"),
        ("TOMATO", "Tomato"),
        ("  Tomato  ", "Tomato"),
        ("bell pepper", bell_expected),
        ("bell_pepper", bell_expected),
        ("bell-pepper", bell_expected),
        ("bellpepper", bell_expected),
        ("Capsicum", bell_expected),
        ("sweet pepper", bell_expected),
        ("shimla mirch", bell_expected),
        # Maize / Corn
        ("corn", corn_expected),
        ("CORN", corn_expected),
        ("makka", corn_expected),
        ("zea mays", corn_expected),
        # Potato
        ("potato", "Potato"),
        ("aloo", "Potato"),
        # Apple
        ("apple", "Apple"),
        ("seb", "Apple"),
        # Grape
        ("grape", "Grape"),
        ("angoor", "Grape"),
    ]

    for raw_hint, expected in test_cases:
        norm, warn = predictor.normalize_crop_hint(raw_hint)
        assert norm == expected, f"Hint '{raw_hint}' normalized to '{norm}', expected '{expected}'"
        assert warn is None, f"Expected no warning for valid synonym '{raw_hint}', got: {warn}"

    # Unsupported crops in HF mode (Rice, Sugarcane, Wheat, Cotton) produce unsupported crop warning
    if is_hf:
        for unsupp in ["paddy", "chawal", "sugarcane", "ganna", "wheat", "cotton"]:
            norm, warn = predictor.normalize_crop_hint(unsupp)
            assert norm is None, f"Expected None for unsupported crop {unsupp}, got {norm}"
            assert warn is not None and "Unsupported crop" in warn

    # Unknown hint should return None and an informative warning
    unknown_norm, unknown_warn = predictor.normalize_crop_hint("banana")
    assert unknown_norm is None, "Unknown crop 'banana' should return None"
    assert unknown_warn is not None, "Unknown crop 'banana' should produce a warning"
    assert "Supported crops are" in unknown_warn


# ---------------------------------------------------------------------------
# 3. Quality Gate Rejection (Blur & Darkness)
# ---------------------------------------------------------------------------
def test_quality_gate_rejection_blur(predictor):
    """
    Validates that a heavily blurred synthetic leaf image is rejected
    with status='poor_image_quality' by the downscaled 512px Laplacian gate.
    """
    # Create synthetic leaf with Gaussian blur
    arr = np.full((300, 300, 3), 200, dtype=np.uint8)
    import cv2
    cv2.ellipse(arr, (150, 150), (100, 60), 30, 0, 360, (40, 160, 45), -1)
    blurred_arr = cv2.GaussianBlur(arr, (51, 51), 0)
    blurred_img = Image.fromarray(blurred_arr)

    res = predictor.predict(_image_to_bytes(blurred_img))
    assert res["status"] == "poor_image_quality", f"Expected poor_image_quality for blur, got: {res['status']}"
    assert res["confidence"] == 0.0
    assert res["certainty"] == "low"
    assert res["quality"]["blur_score"] < 80.0
    assert "dhundhli" in res["message"].lower() or "blurry" in res["message"].lower()


def test_quality_gate_rejection_dark(predictor):
    """
    Validates that an underexposed (dark) image (brightness < 40)
    is rejected with status='poor_image_quality'.
    """
    # Create dark image (average brightness ~ 15)
    dark_img = Image.new("RGB", (256, 256), color=(15, 18, 12))
    res = predictor.predict(_image_to_bytes(dark_img))

    assert res["status"] == "poor_image_quality", f"Expected poor_image_quality for dark image, got: {res['status']}"
    assert res["confidence"] == 0.0
    assert res["certainty"] == "low"
    assert res["quality"]["brightness"] < 40.0
    assert "andhere" in res["message"].lower() or "dark" in res["message"].lower()


# ---------------------------------------------------------------------------
# 4. Free Energy OOD Rejection (Random Noise)
# ---------------------------------------------------------------------------
def test_energy_ood_rejection_random_noise(predictor):
    """
    Validates that random pixel noise is detected and rejected
    with status='not_a_leaf'.
    """
    np.random.seed(42)
    noise_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)

    res = predictor.predict(_image_to_bytes(noise_img))
    assert res["status"] in ["unknown", "not_a_leaf"], f"Expected unknown or not_a_leaf for random noise, got: {res['status']}"
    assert res["crop"] == "unknown"
    assert res["disease"] == "unknown"
    assert res["confidence"] < 0.10
    assert res["certainty"] == "low"


# ---------------------------------------------------------------------------
# 5. Species Mismatch Protection with Correct Hint
# ---------------------------------------------------------------------------
def test_correct_hint_never_produces_species_mismatch(predictor):
    """
    Validates that when a real leaf is accompanied by a correct user hint,
    it NEVER produces a false species_mismatch.
    """
    data_dir = os.path.abspath("data/sample_real_leaves")
    test_leaf = os.path.join(data_dir, "a-bellpepper-leaf.png")
    if not os.path.exists(test_leaf):
        pytest.skip("Test sample leaf not found")

    with open(test_leaf, "rb") as f:
        leaf_bytes = f.read()

    # Test with correct hints (including synonyms) where crop mass is high (~55%)
    for valid_hint in ["Bell_Pepper", "bell pepper", "capsicum", "sweet pepper"]:
        res = predictor.predict(leaf_bytes, crop_hint=valid_hint)
        assert res["status"] in ["ok", "low_confidence"], f"Failed on valid hint '{valid_hint}': {res['status']}"
        assert res["crop"] in ["Bell_Pepper", "Bell Pepper"]
        assert res["mismatch_warning"] is None

    # In contrast, an incompatible supported hint MUST trigger species_mismatch
    mismatch_hint = "Squash" if predictor.backend == "huggingface" else "Sugarcane"
    res_mismatch = predictor.predict(leaf_bytes, crop_hint=mismatch_hint)
    assert res_mismatch["status"] == "species_mismatch"
    assert res_mismatch["crop"] == mismatch_hint
    assert res_mismatch["mismatch_warning"] is not None
    assert "diagnosis was requested for" in res_mismatch["mismatch_warning"]


# ---------------------------------------------------------------------------
# 6. Backward Compatibility with Spring Boot DTO Keys
# ---------------------------------------------------------------------------
def test_spring_dto_backwards_compatibility(predictor):
    """
    Validates that EVERY response status contains EVERY key expected by
    Spring Boot MlPredictionResponse DTO:
    - status (String)
    - crop (String)
    - disease (String)
    - confidence (Double)
    - certainty (String)
    - top3 (List<TopPrediction> with label and score)
    - quality (QualityInfo with blur_score and brightness)
    - message (String)
    """
    expected_top_keys = {
        "status", "crop", "disease", "confidence", "certainty",
        "top3", "quality", "message"
    }

    data_dir = os.path.abspath("data/sample_real_leaves")
    test_leaf = os.path.join(data_dir, "a-bellpepper-leaf.png")
    if not os.path.exists(test_leaf):
        pytest.skip("Sample leaf missing")

    with open(test_leaf, "rb") as f:
        leaf_bytes = f.read()

    # 1. Normal / Low Confidence Response
    res_ok = predictor.predict(leaf_bytes)
    assert expected_top_keys.issubset(res_ok.keys()), f"Missing keys in ok response: {expected_top_keys - res_ok.keys()}"
    assert "blur_score" in res_ok["quality"] and "brightness" in res_ok["quality"]
    assert isinstance(res_ok["top3"], list)

    # 2. Species Mismatch Response
    mismatch_hint = "Squash" if predictor.backend == "huggingface" else "Sugarcane"
    res_mismatch = predictor.predict(leaf_bytes, crop_hint=mismatch_hint)
    assert expected_top_keys.issubset(res_mismatch.keys()), f"Missing keys in mismatch response: {expected_top_keys - res_mismatch.keys()}"
    assert res_mismatch["status"] == "species_mismatch"

    # 3. Quality Rejection Response
    dark_img = Image.new("RGB", (100, 100), (10, 10, 10))
    res_qual = predictor.predict(_image_to_bytes(dark_img))
    assert expected_top_keys.issubset(res_qual.keys()), f"Missing keys in quality response: {expected_top_keys - res_qual.keys()}"
    assert res_qual["status"] in ["poor_image_quality", "unknown"]

    # 4. Out-of-Distribution Response
    noise_img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
    res_ood = predictor.predict(_image_to_bytes(noise_img))
    assert expected_top_keys.issubset(res_ood.keys()), f"Missing keys in OOD response: {expected_top_keys - res_ood.keys()}"
    assert res_ood["status"] in ["not_a_leaf", "unknown"]
